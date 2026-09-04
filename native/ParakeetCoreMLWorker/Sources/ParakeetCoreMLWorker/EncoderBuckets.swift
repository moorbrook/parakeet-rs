import CoreML
import FluidAudio
import Foundation

/// Managers holding offline encoders compiled at windows shorter than 15 s.
///
/// `UnifiedAsrManager` zero-pads every utterance to its encoder's compiled mel
/// window, so a one-word utterance runs the whole 15 s graph: measured at 25.5
/// of the 35 ms one-second result. Encoder cost is close to linear in that
/// window (see `docs/asr/COMPUTE_PLAN.md`), so a short utterance sent through a
/// shorter encoder pays proportionally less. The Neural Engine has no flexible
/// shapes, so each window is a separate compiled program; a request is routed
/// to the narrowest one that still holds it.
///
/// Buckets are discovered by filename in the model directory rather than
/// configured, so a build with no bucket artifacts behaves exactly as before.
struct EncoderBuckets {
    struct Bucket {
        let seconds: Int
        let sampleCount: Int
        let manager: UnifiedAsrManager
    }

    /// Ascending by window, so the first bucket that fits is the narrowest.
    private let buckets: [Bucket]

    static let empty = EncoderBuckets(buckets: [])

    private init(buckets: [Bucket]) {
        self.buckets = buckets
    }

    var descriptions: [String] {
        buckets.map { "\($0.seconds)s" }
    }

    var isEmpty: Bool { buckets.isEmpty }

    /// The narrowest bucket whose window holds `sampleCount` 16 kHz samples,
    /// or `nil` when no bucket does and the caller should use its 15 s manager.
    func manager(forSampleCount sampleCount: Int) -> UnifiedAsrManager? {
        guard let seconds = Self.select(from: buckets.map(\.seconds), sampleCount: sampleCount)
        else { return nil }
        return buckets.first { $0.seconds == seconds }?.manager
    }

    /// The narrowest window in `windows` that holds `sampleCount` samples at
    /// 16 kHz, or `nil` when none does. Split out from `manager(forSampleCount:)`
    /// so the routing rule can be tested without loading 569 MB of weights.
    static func select(from windows: [Int], sampleCount: Int) -> Int? {
        windows.sorted().first { sampleCount <= $0 * 16_000 }
    }

    /// Every bucket window, in seconds, whose encoder bundle is present in
    /// `directory`. Windows of 15 s and above are not buckets: that is the
    /// stock encoder, which the caller already has.
    static func availableWindows(in directory: URL, precision: UnifiedEncoderPrecision) -> [Int] {
        (1..<15).filter { seconds in
            let name = ModelNames.ParakeetUnified.offlineEncoderFile(
                precision: precision, windowSeconds: seconds)
            return FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(name).path)
        }
    }

    /// Load one manager per requested window. Every model but the encoder is
    /// loaded again per bucket; they are 17 MB together against 569 MB for an
    /// encoder, and sharing them would mean reaching inside the manager.
    static func load(
        windows: [Int],
        directory: URL,
        computeUnits: MLComputeUnits,
        precision: UnifiedEncoderPrecision
    ) async throws -> Self {
        var loaded: [Bucket] = []
        for seconds in windows.sorted() {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            let manager = UnifiedAsrManager(
                configuration: configuration,
                config: UnifiedConfig(offlineWindowSeconds: seconds),
                encoderPrecision: precision
            )
            try await manager.loadModels(from: directory)
            loaded.append(
                Bucket(
                    seconds: seconds,
                    sampleCount: seconds * 16_000,
                    manager: manager
                ))
        }
        return Self(buckets: loaded)
    }
}
