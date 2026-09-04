import CoreML
import Foundation
import ObjectiveC

/// Per-utterance Core ML dispatch accounting for the Parakeet Unified pipeline.
///
/// The decode pipeline lives inside FluidAudio's `UnifiedAsrManager` and
/// `UnifiedRnntDecoder`, which are a pinned external dependency. Rather than
/// vendoring or patching that package, this profiler replaces `MLModel`'s
/// prediction implementations with timing wrappers that call straight through.
/// Every stage boundary is then derived from the resulting dispatch timeline:
///
/// - mel: the gap between the previous dispatch (or decode start) and the
///   encoder dispatch that consumes its output, per 15 s window;
/// - encoder: the encoder dispatch itself;
/// - decoder loop: everything from an encoder dispatch's end to the last
///   dispatch belonging to that window;
/// - post: the tail after the final dispatch (tokenizer decode, overlap merge).
///
/// Dispatches are attributed to a stage by the input feature names FluidAudio's
/// providers declare, so the classification does not depend on holding a
/// reference to any of the three `MLModel` instances.
///
/// Installation is opt-in (`--emit-stage-timings`). The shipping dictation path
/// never installs it.
final class StageProfiler: @unchecked Sendable {
    static let shared = StageProfiler()

    enum Stage: UInt8 {
        case encoder
        case decoder
        case joint
        case other
    }

    struct Event {
        let stage: Stage
        let startNanoseconds: UInt64
        let endNanoseconds: UInt64
    }

    /// Timeline-derived stage breakdown for one `transcribe` call.
    struct Report: Encodable {
        /// 48 kHz → 16 kHz conversion, measured by the worker before decoding.
        /// Set by the caller since it happens outside the profiled interval.
        var resampleMs: Double = 0
        let windows: Int
        let encoderCalls: Int
        let decoderCalls: Int
        let jointCalls: Int
        let otherCalls: Int
        let melMs: Double
        let encoderMs: Double
        let decodeLoopMs: Double
        let decodeLoopDispatchMs: Double
        let decoderDispatchMs: Double
        let jointDispatchMs: Double
        let postMs: Double
        let totalMs: Double
        /// `MLModelConfiguration.computeUnits` read off the live model object
        /// each stage dispatched to, e.g. "encoder=cpu-and-neural-engine
        /// decoder=cpu-only joint=cpu-only".
        let computeUnits: String
    }

    /// Guards `events` and `installed`. Prediction dispatch is serial in the
    /// offline path, so an uncontended lock costs a few tens of nanoseconds
    /// against a dispatch floor three orders of magnitude larger.
    private let lock = NSLock()
    private var events: [Event] = []
    private var wrappedMethods: Set<String> = []
    private var recording = false
    /// Compute units observed on the live `MLModel` each stage dispatched to.
    private var observedComputeUnits: [Stage: MLComputeUnits] = [:]

    /// Cap the timeline so a pathological input cannot grow it without bound.
    /// A 20 s utterance produces roughly 1,500 dispatches.
    private static let maximumEvents = 200_000

    private static let depthKey = "com.parakeet.stage-profiler.depth"

    private init() {}

    // MARK: - Clock

    static func now() -> UInt64 {
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    private static func milliseconds(_ nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000
    }

    // MARK: - Recording

    /// Begin a fresh timeline. Returns the start timestamp to anchor it.
    func beginUtterance() -> UInt64 {
        // Core ML registers some engine classes only once a prediction has run,
        // so the sweep repeats until all three stages have been seen. It walks
        // the whole Objective-C class list, which costs about 0.8 ms, so it
        // stops as soon as the pipeline is fully covered.
        if needsSweep() { install() }
        lock.lock()
        events.removeAll(keepingCapacity: true)
        recording = true
        lock.unlock()
        return Self.now()
    }

    /// Close the timeline and reduce it to per-stage durations.
    func endUtterance(startNanoseconds: UInt64, endNanoseconds: UInt64) -> Report {
        lock.lock()
        recording = false
        let timeline = events.sorted { $0.startNanoseconds < $1.startNanoseconds }
        events.removeAll(keepingCapacity: true)
        let placements = observedComputeUnits
        lock.unlock()
        return Self.reduce(
            timeline: timeline,
            startNanoseconds: startNanoseconds,
            endNanoseconds: endNanoseconds,
            computeUnits: Self.describe(placements)
        )
    }

    /// `PARAKEET_STAGE_TRACE=1` prints every intercepted dispatch with its
    /// class and input feature names. This is the diagnostic that shows which
    /// Core ML entry point a stage actually uses, which matters because the
    /// encoder's asynchronous path does not go through the same selector as the
    /// synchronous decoder and joint calls.
    static let trace = ProcessInfo.processInfo.environment["PARAKEET_STAGE_TRACE"] != nil

    static func traceEntry(_ model: AnyObject, _ input: MLFeatureProvider, _ kind: String) {
        guard trace else { return }
        FileHandle.standardError.write(
            Data(
                "entry kind=\(kind) class=\(NSStringFromClass(type(of: model))) names=\(input.featureNames.sorted().joined(separator: "+"))\n"
                    .utf8))
    }

    fileprivate func record(_ event: Event, model: AnyObject) {
        if Self.trace {
            FileHandle.standardError.write(
                Data(
                    "trace stage=\(Self.name(of: event.stage)) class=\(NSStringFromClass(type(of: model)))\n"
                        .utf8))
        }
        let placement = (model as? MLModel)?.configuration.computeUnits
        lock.lock()
        if recording, events.count < Self.maximumEvents {
            events.append(event)
        }
        if let placement {
            observedComputeUnits[event.stage] = placement
        }
        lock.unlock()
    }

    private static func describe(_ placements: [Stage: MLComputeUnits]) -> String {
        [Stage.encoder, .decoder, .joint]
            .compactMap { stage -> String? in
                guard let placement = placements[stage] else { return nil }
                return "\(name(of: stage))=\(name(of: placement))"
            }
            .joined(separator: " ")
    }

    private static func name(of stage: Stage) -> String {
        switch stage {
        case .encoder: "encoder"
        case .decoder: "decoder"
        case .joint: "joint"
        case .other: "other"
        }
    }

    private static func name(of computeUnits: MLComputeUnits) -> String {
        switch computeUnits {
        case .cpuOnly: "cpu-only"
        case .cpuAndGPU: "cpu-and-gpu"
        case .cpuAndNeuralEngine: "cpu-and-neural-engine"
        case .all: "all"
        @unknown default: "unknown"
        }
    }

    // MARK: - Reduction

    static func reduce(
        timeline: [Event],
        startNanoseconds: UInt64,
        endNanoseconds: UInt64,
        computeUnits: String
    ) -> Report {
        var encoderCalls = 0
        var decoderCalls = 0
        var jointCalls = 0
        var otherCalls = 0
        var encoderNanoseconds: UInt64 = 0
        var decoderNanoseconds: UInt64 = 0
        var jointNanoseconds: UInt64 = 0
        for event in timeline {
            let elapsed = event.endNanoseconds &- event.startNanoseconds
            switch event.stage {
            case .encoder:
                encoderCalls += 1
                encoderNanoseconds &+= elapsed
            case .decoder:
                decoderCalls += 1
                decoderNanoseconds &+= elapsed
            case .joint:
                jointCalls += 1
                jointNanoseconds &+= elapsed
            case .other:
                otherCalls += 1
            }
        }

        // Mel is the work between the end of the previous dispatch (or the
        // start of the utterance) and the encoder dispatch that consumes it.
        // The decode loop is what follows an encoder dispatch until the last
        // dispatch of that window.
        var melNanoseconds: UInt64 = 0
        var decodeLoopNanoseconds: UInt64 = 0
        var previousEnd = startNanoseconds
        var openEncoderEnd: UInt64?
        var lastEnd = startNanoseconds
        for event in timeline {
            if event.stage == .encoder {
                if let encoderEnd = openEncoderEnd {
                    decodeLoopNanoseconds &+= lastEnd &- encoderEnd
                }
                melNanoseconds &+= event.startNanoseconds &- min(previousEnd, event.startNanoseconds)
                openEncoderEnd = event.endNanoseconds
            }
            previousEnd = max(previousEnd, event.endNanoseconds)
            lastEnd = max(lastEnd, event.endNanoseconds)
        }
        if let encoderEnd = openEncoderEnd {
            decodeLoopNanoseconds &+= lastEnd &- encoderEnd
        }

        return Report(
            windows: encoderCalls,
            encoderCalls: encoderCalls,
            decoderCalls: decoderCalls,
            jointCalls: jointCalls,
            otherCalls: otherCalls,
            melMs: milliseconds(melNanoseconds),
            encoderMs: milliseconds(encoderNanoseconds),
            decodeLoopMs: milliseconds(decodeLoopNanoseconds),
            decodeLoopDispatchMs: milliseconds(decoderNanoseconds &+ jointNanoseconds),
            decoderDispatchMs: milliseconds(decoderNanoseconds),
            jointDispatchMs: milliseconds(jointNanoseconds),
            postMs: milliseconds(endNanoseconds &- max(lastEnd, startNanoseconds)),
            totalMs: milliseconds(endNanoseconds &- startNanoseconds),
            computeUnits: computeUnits
        )
    }

    // MARK: - Classification

    /// Map a prediction input to the pipeline stage that produced it.
    ///
    /// FluidAudio's three feature providers declare disjoint feature-name sets
    /// (`UnifiedFeatureProviders.swift`), so the input alone identifies the model.
    static func classify(_ input: MLFeatureProvider) -> Stage {
        let names = input.featureNames
        if names.contains("mel") { return .encoder }
        if names.contains("encoder_step") { return .joint }
        if names.contains("targets") { return .decoder }
        return .other
    }

    // MARK: - Installation

    /// Replace the prediction entry points of `MLModel` and every registered
    /// subclass with timing wrappers.
    ///
    /// Swizzling `MLModel` alone is not enough: Core ML's asynchronous entry
    /// point is overridden by a private subclass that never calls up to
    /// `MLModel`, so an `MLModel`-only wrapper silently reports zero encoder
    /// dispatches. Every class that supplies its own implementation of one of
    /// the four selectors is wrapped, which is why this must run *after* the
    /// models are loaded and their classes registered.
    ///
    /// Idempotent, and reports what was wrapped so a future SDK that renames or
    /// removes a selector fails loudly in the bench log instead of quietly
    /// under-counting.
    @discardableResult
    func install() -> [String] {
        var wrapped: [String] = []
        for target in Self.predictionClasses() {
            let name = NSStringFromClass(target)
            for selector in ["predictionFromFeatures:error:", "predictionFromFeatures:options:error:"]
            where Self.implementsItself(target, NSSelectorFromString(selector))
                && claim("\(name) \(selector)")
            {
                if installSynchronous(
                    selector, on: target, hasOptions: selector.contains("options")
                ) {
                    wrapped.append("\(name) \(selector)")
                }
            }
            let submit = "submitPredictionRequest:completionHandler:"
            if Self.implementsItself(target, NSSelectorFromString(submit)), claim("\(name) \(submit)"),
                installRequest(submit, on: target)
            {
                wrapped.append("\(name) \(submit)")
            }
            for selector in [
                "predictionFromFeatures:completionHandler:",
                "predictionFromFeatures:options:completionHandler:",
            ]
            where Self.implementsItself(target, NSSelectorFromString(selector))
                && claim("\(name) \(selector)")
            {
                if installAsynchronous(
                    selector, on: target, hasOptions: selector.contains("options")
                ) {
                    wrapped.append("\(name) \(selector)")
                }
            }
        }
        return wrapped
    }

    /// Reserve a class/selector pair, returning false if it is already wrapped.
    /// Core ML registers some engine classes only once a prediction has run, so
    /// the sweep repeats before each utterance and must not double-wrap.
    /// Whether any pipeline stage has yet to be observed dispatching.
    private func needsSweep() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return ![Stage.encoder, .decoder, .joint].allSatisfy(observedComputeUnits.keys.contains)
    }

    private func claim(_ key: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return wrappedMethods.insert(key).inserted
    }

    /// `MLModel` plus every registered subclass of it.
    private static func predictionClasses() -> [AnyClass] {
        let count = objc_getClassList(nil, 0)
        guard count > 0 else { return [MLModel.self] }
        let buffer = UnsafeMutablePointer<AnyClass>.allocate(capacity: Int(count))
        defer { buffer.deallocate() }
        let autoreleasing = AutoreleasingUnsafeMutablePointer<AnyClass>(buffer)
        let written = Int(objc_getClassList(autoreleasing, count))

        let root = ObjectIdentifier(MLModel.self)
        var classes: [AnyClass] = [MLModel.self]
        for index in 0..<written {
            let candidate: AnyClass = buffer[index]
            guard ObjectIdentifier(candidate) != root else { continue }
            var walker: AnyClass? = class_getSuperclass(candidate)
            while let current = walker {
                if ObjectIdentifier(current) == root {
                    classes.append(candidate)
                    break
                }
                walker = class_getSuperclass(current)
            }
        }
        return classes
    }

    /// True when `target` supplies its own implementation of `selector` rather
    /// than inheriting one — the test that keeps a subclass sweep from
    /// wrapping the same inherited implementation many times over.
    private static func implementsItself(_ target: AnyClass, _ selector: Selector) -> Bool {
        guard let method = class_getInstanceMethod(target, selector) else { return false }
        if ObjectIdentifier(target) == ObjectIdentifier(MLModel.self) { return true }
        guard let superclass = class_getSuperclass(target) else { return true }
        guard let inherited = class_getInstanceMethod(superclass, selector) else { return true }
        return method != inherited
    }

    /// Re-entrancy depth for the current thread.
    ///
    /// Core ML's convenience entry points call one another, so the same logical
    /// prediction can pass through two wrapped selectors. Only the outermost
    /// one is recorded; otherwise every dispatch would be counted twice.
    private static func enter() -> Bool {
        let dictionary = Thread.current.threadDictionary
        let depth = (dictionary[depthKey] as? Int) ?? 0
        dictionary[depthKey] = depth + 1
        return depth == 0
    }

    private static func leave() {
        let dictionary = Thread.current.threadDictionary
        let depth = (dictionary[depthKey] as? Int) ?? 1
        dictionary[depthKey] = depth - 1
    }

    private func installSynchronous(_ name: String, on target: AnyClass, hasOptions: Bool) -> Bool {
        let selector = NSSelectorFromString(name)
        guard let method = class_getInstanceMethod(target, selector) else { return false }
        let profiler = self
        let originalBox = ImplementationBox()

        let implementation: IMP
        if hasOptions {
            typealias Original = @convention(c) (
                AnyObject, Selector, MLFeatureProvider, MLPredictionOptions,
                AutoreleasingUnsafeMutablePointer<NSError?>?
            ) -> AnyObject?
            let block: @convention(block) (
                AnyObject, MLFeatureProvider, MLPredictionOptions,
                AutoreleasingUnsafeMutablePointer<NSError?>?
            ) -> AnyObject? = { object, input, options, error in
                Self.traceEntry(object, input, name)
                let outermost = Self.enter()
                defer { Self.leave() }
                let original = unsafeBitCast(originalBox.value, to: Original.self)
                guard outermost else { return original(object, selector, input, options, error) }
                let start = Self.now()
                let result = original(object, selector, input, options, error)
                profiler.record(
                    Event(
                        stage: Self.classify(input), startNanoseconds: start,
                        endNanoseconds: Self.now()
                    ),
                    model: object
                )
                return result
            }
            implementation = imp_implementationWithBlock(block)
        } else {
            typealias Original = @convention(c) (
                AnyObject, Selector, MLFeatureProvider,
                AutoreleasingUnsafeMutablePointer<NSError?>?
            ) -> AnyObject?
            let block: @convention(block) (
                AnyObject, MLFeatureProvider, AutoreleasingUnsafeMutablePointer<NSError?>?
            ) -> AnyObject? = { object, input, error in
                Self.traceEntry(object, input, name)
                let outermost = Self.enter()
                defer { Self.leave() }
                let original = unsafeBitCast(originalBox.value, to: Original.self)
                guard outermost else { return original(object, selector, input, error) }
                let start = Self.now()
                let result = original(object, selector, input, error)
                profiler.record(
                    Event(
                        stage: Self.classify(input), startNanoseconds: start,
                        endNanoseconds: Self.now()
                    ),
                    model: object
                )
                return result
            }
            implementation = imp_implementationWithBlock(block)
        }
        originalBox.value = method_setImplementation(method, implementation)
        return true
    }

    private func installAsynchronous(_ name: String, on target: AnyClass, hasOptions: Bool) -> Bool {
        let selector = NSSelectorFromString(name)
        guard let method = class_getInstanceMethod(target, selector) else { return false }
        let profiler = self
        let originalBox = ImplementationBox()

        let implementation: IMP
        if hasOptions {
            typealias Original = @convention(c) (
                AnyObject, Selector, MLFeatureProvider, MLPredictionOptions,
                @escaping (MLFeatureProvider?, Error?) -> Void
            ) -> Void
            let block:
                @convention(block) (
                    AnyObject, MLFeatureProvider, MLPredictionOptions,
                    @escaping (MLFeatureProvider?, Error?) -> Void
                ) -> Void = { object, input, options, completion in
                    Self.traceEntry(object, input, name)
                    let original = unsafeBitCast(originalBox.value, to: Original.self)
                    let start = Self.now()
                    let stage = Self.classify(input)
                    original(object, selector, input, options) { output, error in
                        profiler.record(
                            Event(
                                stage: stage, startNanoseconds: start, endNanoseconds: Self.now()
                            ),
                            model: object
                        )
                        completion(output, error)
                    }
                }
            implementation = imp_implementationWithBlock(block)
        } else {
            typealias Original = @convention(c) (
                AnyObject, Selector, MLFeatureProvider,
                @escaping (MLFeatureProvider?, Error?) -> Void
            ) -> Void
            let block:
                @convention(block) (
                    AnyObject, MLFeatureProvider, @escaping (MLFeatureProvider?, Error?) -> Void
                ) -> Void = { object, input, completion in
                    Self.traceEntry(object, input, name)
                    let original = unsafeBitCast(originalBox.value, to: Original.self)
                    let start = Self.now()
                    let stage = Self.classify(input)
                    original(object, selector, input) { output, error in
                        profiler.record(
                            Event(
                                stage: stage, startNanoseconds: start, endNanoseconds: Self.now()
                            ),
                            model: object
                        )
                        completion(output, error)
                    }
                }
            implementation = imp_implementationWithBlock(block)
        }
        originalBox.value = method_setImplementation(method, implementation)
        return true
    }
}

extension StageProfiler {
    /// Wrap Core ML's request-object prediction entry point.
    ///
    /// Swift's `MLModel.prediction(from:) async` does not reach any of the
    /// `predictionFromFeatures:` selectors: it submits an
    /// `MLPredictionRequest`-style object instead. That is the path the encoder
    /// takes, so without this the encoder's dispatch is invisible while the
    /// synchronous decoder and joint calls are both captured.
    ///
    /// The request object's input provider is reached by key-value coding
    /// because the request class is private; the first property that yields an
    /// `MLFeatureProvider` is cached per request class.
    fileprivate func installRequest(_ name: String, on target: AnyClass) -> Bool {
        let selector = NSSelectorFromString(name)
        guard let method = class_getInstanceMethod(target, selector) else { return false }
        let profiler = self
        let originalBox = ImplementationBox()

        typealias Original = @convention(c) (
            AnyObject, Selector, AnyObject, @escaping (AnyObject?, Error?) -> Void
        ) -> Void
        let block:
            @convention(block) (AnyObject, AnyObject, @escaping (AnyObject?, Error?) -> Void) ->
                Void = { object, request, completion in
                    let input = Self.inputProvider(of: request)
                    if let input { Self.traceEntry(object, input, name) }
                    let original = unsafeBitCast(originalBox.value, to: Original.self)
                    let start = Self.now()
                    let stage = input.map(Self.classify) ?? .other
                    original(object, selector, request) { response, error in
                        profiler.record(
                            Event(stage: stage, startNanoseconds: start, endNanoseconds: Self.now()),
                            model: object
                        )
                        completion(response, error)
                    }
                }
        originalBox.value = method_setImplementation(method, imp_implementationWithBlock(block))
        return true
    }

    /// The `MLFeatureProvider` carried by a private prediction-request object.
    fileprivate static func inputProvider(of request: AnyObject) -> MLFeatureProvider? {
        if let direct = request as? MLFeatureProvider { return direct }
        let requestClass: AnyClass = type(of: request)
        let key = ObjectIdentifier(requestClass)
        if let cached = providerKeys.withLock({ $0[key] }) {
            return (request as? NSObject)?.value(forKey: cached) as? MLFeatureProvider
        }
        guard let object = request as? NSObject else { return nil }
        var count: UInt32 = 0
        guard let properties = class_copyPropertyList(requestClass, &count) else { return nil }
        defer { free(properties) }
        for index in 0..<Int(count) {
            let propertyName = String(cString: property_getName(properties[index]))
            guard let value = object.value(forKey: propertyName) as? MLFeatureProvider else {
                continue
            }
            providerKeys.withLock { $0[key] = propertyName }
            return value
        }
        return nil
    }
}

/// Cached KVC key per prediction-request class.
private let providerKeys = Locked<[ObjectIdentifier: String]>([:])

private final class Locked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<Result>(_ body: (inout Value) -> Result) -> Result {
        lock.lock()
        defer { lock.unlock() }
        return body(&value)
    }
}

/// Holds the replaced implementation so the installing closure can capture it
/// before `method_setImplementation` returns it.
private final class ImplementationBox: @unchecked Sendable {
    var value: IMP!
}
