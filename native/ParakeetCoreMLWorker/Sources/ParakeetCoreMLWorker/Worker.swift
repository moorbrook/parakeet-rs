import CoreML
import FluidAudio
import Foundation

private let protocolMagic = Data([0x50, 0x52, 0x4b, 0x54]) // "PRKT"
private let protocolVersion: UInt32 = 1
private let requestHeaderBytes = 16
private let maximumAudioSeconds: UInt64 = 30 * 60
private let maximumLongRegimeSeconds: UInt32 = 60
private let maximumResponseBytes = 4 * 1024 * 1024

private enum WorkerError: LocalizedError {
    case invalidArgument(String)
    case truncatedRequest
    case invalidMagic
    case unsupportedVersion(UInt32)
    case invalidSampleRate(UInt32)
    case invalidSampleCount(UInt32)
    case oversizedResponse(Int)

    var errorDescription: String? {
        switch self {
        case .invalidArgument(let message): message
        case .truncatedRequest: "request ended before the declared audio payload"
        case .invalidMagic: "request magic is not PRKT"
        case .unsupportedVersion(let version): "unsupported protocol version \(version)"
        case .invalidSampleRate(let rate): "invalid sample rate \(rate)"
        case .invalidSampleCount(let count): "invalid sample count \(count)"
        case .oversizedResponse(let bytes): "response is too large (\(bytes) bytes)"
        }
    }
}

private struct WorkerOptions {
    let modelDirectory: URL?
    let modelRoot: URL?
    let shortComputeUnits: MLComputeUnits
    let longComputeUnits: MLComputeUnits
    let longRegimeSeconds: UInt32
    let emitStageTimings: Bool
    /// Run the RNNT prediction network and joint natively instead of through a
    /// CoreML dispatch per step. On by default; the CoreML path stays reachable
    /// so the two can be compared on the same build.
    let nativeRnnt: Bool
    /// Short-window encoder buckets: `nil` means discover them in the model
    /// directory, an empty array disables them, and an explicit list is used
    /// verbatim.
    let encoderBuckets: [Int]?

    static func parse(_ arguments: [String]) throws -> Self {
        var modelDirectory: URL?
        var modelRoot: URL?
        var shortComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
        var longComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
        var longRegimeSeconds: UInt32 = 8
        var emitStageTimings = false
        var nativeRnnt = true
        var encoderBuckets: [Int]?
        var index = 0
        while index < arguments.count {
            switch arguments[index] {
            case "--model-dir":
                index += 1
                guard index < arguments.count else {
                    throw WorkerError.invalidArgument("--model-dir needs a path")
                }
                modelDirectory = URL(fileURLWithPath: arguments[index], isDirectory: true)
            case "--model-root":
                index += 1
                guard index < arguments.count else {
                    throw WorkerError.invalidArgument("--model-root needs a path")
                }
                modelRoot = URL(fileURLWithPath: arguments[index], isDirectory: true)
            case "--compute-units":
                index += 1
                guard index < arguments.count else {
                    throw WorkerError.invalidArgument("--compute-units needs a name")
                }
                let computeUnits = try parseComputeUnits(arguments[index])
                shortComputeUnits = computeUnits
                longComputeUnits = computeUnits
            case "--short-compute-units":
                index += 1
                guard index < arguments.count else {
                    throw WorkerError.invalidArgument("--short-compute-units needs a name")
                }
                shortComputeUnits = try parseComputeUnits(arguments[index])
            case "--long-compute-units":
                index += 1
                guard index < arguments.count else {
                    throw WorkerError.invalidArgument("--long-compute-units needs a name")
                }
                longComputeUnits = try parseComputeUnits(arguments[index])
            case "--long-regime-seconds":
                index += 1
                guard index < arguments.count,
                    let seconds = UInt32(arguments[index]),
                    (1...maximumLongRegimeSeconds).contains(seconds)
                else {
                    throw WorkerError.invalidArgument(
                        "--long-regime-seconds must be between 1 and 60"
                    )
                }
                longRegimeSeconds = seconds
            case "--encoder-buckets":
                index += 1
                guard index < arguments.count else {
                    throw WorkerError.invalidArgument(
                        "--encoder-buckets needs auto, none, or a comma-separated list of seconds"
                    )
                }
                encoderBuckets = try parseEncoderBuckets(arguments[index])
            case "--emit-stage-timings":
                emitStageTimings = true
            case "--rnnt-engine":
                index += 1
                guard index < arguments.count,
                    ["native", "coreml"].contains(arguments[index])
                else {
                    throw WorkerError.invalidArgument("--rnnt-engine must be native or coreml")
                }
                nativeRnnt = arguments[index] == "native"
            case "-h", "--help":
                let usage =
                    "usage: parakeet-coreml-worker [--model-dir DIR | --model-root DIR] "
                    + "[--compute-units NAME | --short-compute-units NAME "
                    + "--long-compute-units NAME --long-regime-seconds N] "
                    + "[--encoder-buckets auto|none|N,N,...] [--rnnt-engine native|coreml] "
                    + "[--emit-stage-timings]\n"
                FileHandle.standardError.write(
                    Data(usage.utf8)
                )
                Foundation.exit(0)
            default:
                throw WorkerError.invalidArgument("unknown argument: \(arguments[index])")
            }
            index += 1
        }
        guard modelDirectory == nil || modelRoot == nil else {
            throw WorkerError.invalidArgument("--model-dir and --model-root are mutually exclusive")
        }
        return Self(
            modelDirectory: modelDirectory,
            modelRoot: modelRoot,
            shortComputeUnits: shortComputeUnits,
            longComputeUnits: longComputeUnits,
            longRegimeSeconds: longRegimeSeconds,
            emitStageTimings: emitStageTimings,
            nativeRnnt: nativeRnnt,
            encoderBuckets: encoderBuckets
        )
    }

    private static func parseEncoderBuckets(_ value: String) throws -> [Int]? {
        switch value {
        case "auto": return nil
        case "none": return []
        default:
            let seconds = try value.split(separator: ",").map { field -> Int in
                guard let parsed = Int(field), (1..<15).contains(parsed) else {
                    throw WorkerError.invalidArgument(
                        "--encoder-buckets entries must be whole seconds between 1 and 14"
                    )
                }
                return parsed
            }
            return Array(Set(seconds)).sorted()
        }
    }

    private static func parseComputeUnits(_ value: String) throws -> MLComputeUnits {
        switch value {
        case "all": .all
        case "cpu-and-gpu": .cpuAndGPU
        case "cpu-and-neural-engine": .cpuAndNeuralEngine
        case "cpu-only": .cpuOnly
        default:
            throw WorkerError.invalidArgument(
                "compute units must be all, cpu-and-gpu, cpu-and-neural-engine, or cpu-only"
            )
        }
    }
}

/// One RNNT emission on the request's own timeline. `text` is the detokenized
/// piece with the SentencePiece word-start marker already rendered as a leading
/// space, so the Rust side can group tokens into words without a vocabulary.
/// Rust's Hold-mode window merge aligns neighbouring windows on these.
private struct TokenSpan: Encodable {
    let text: String
    let startS: Double
    let endS: Double
}

private struct WorkerResponse: Encodable {
    let kind: String
    let ok: Bool
    let text: String?
    let error: String?
    let loadSeconds: Double?
    let decodeSeconds: Double?
    let resampleSeconds: Double?
    let stages: StageProfiler.Report?
    let tokenSpans: [TokenSpan]?

    static func ready(loadSeconds: Double) -> Self {
        Self(
            kind: "ready",
            ok: true,
            text: nil,
            error: nil,
            loadSeconds: loadSeconds,
            decodeSeconds: nil,
            resampleSeconds: nil,
            stages: nil,
            tokenSpans: nil
        )
    }

    static func result(
        text: String,
        decodeSeconds: Double,
        resampleSeconds: Double,
        stages: StageProfiler.Report?,
        tokenSpans: [TokenSpan]
    ) -> Self {
        Self(
            kind: "result",
            ok: true,
            text: text,
            error: nil,
            loadSeconds: nil,
            decodeSeconds: decodeSeconds,
            resampleSeconds: resampleSeconds,
            stages: stages,
            tokenSpans: tokenSpans
        )
    }

    static func failure(kind: String, error: Error) -> Self {
        Self(
            kind: kind,
            ok: false,
            text: nil,
            error: error.localizedDescription,
            loadSeconds: nil,
            decodeSeconds: nil,
            resampleSeconds: nil,
            stages: nil,
            tokenSpans: nil
        )
    }
}

@main
private struct ParakeetCoreMLWorker {
    static func main() async {
        do {
            let options = try WorkerOptions.parse(Array(CommandLine.arguments.dropFirst()))
            try await run(options: options)
        } catch {
            try? writeResponse(.failure(kind: "fatal", error: error))
            FileHandle.standardError.write(Data("parakeet-coreml-worker: \(error)\n".utf8))
            Foundation.exit(1)
        }
    }

    private static func run(options: WorkerOptions) async throws {
        let loadStart = ContinuousClock.now
        let shortManager = try await loadManager(
            computeUnits: options.shortComputeUnits,
            options: options
        )
        let longManager: UnifiedAsrManager?
        if options.longComputeUnits == options.shortComputeUnits {
            longManager = nil
        } else {
            longManager = try await loadManager(
                computeUnits: options.longComputeUnits,
                options: options
            )
        }
        let modelDirectory =
            options.modelDirectory
            ?? options.modelRoot?
                .appendingPathComponent(Repo.parakeetUnified.folderName, isDirectory: true)
        var buckets = EncoderBuckets.empty
        if let modelDirectory {
            let windows =
                options.encoderBuckets
                ?? EncoderBuckets.availableWindows(in: modelDirectory, precision: .int8)
            if !windows.isEmpty {
                // Bucket encoders are short by construction, so they follow the
                // short regime's compute units.
                buckets = try await EncoderBuckets.load(
                    windows: windows,
                    directory: modelDirectory,
                    computeUnits: options.shortComputeUnits,
                    precision: .int8,
                    rnntDecoderFactory: rnntDecoderFactory(options)
                )
            }
        }
        let loadSeconds = seconds(since: loadStart)
        if !buckets.isEmpty {
            let windows = buckets.descriptions.joined(separator: ", ")
            FileHandle.standardError.write(
                Data("parakeet-coreml-worker: encoder buckets \(windows)\n".utf8))
        }

        // Installed after loading so Core ML's private MLModel subclasses are
        // registered and can be wrapped; see StageProfiler.install().
        if options.emitStageTimings {
            let wrapped = StageProfiler.shared.install()
            guard !wrapped.isEmpty else {
                throw WorkerError.invalidArgument(
                    "--emit-stage-timings found no MLModel prediction selector to wrap"
                )
            }
            FileHandle.standardError.write(
                Data(
                    "parakeet-coreml-worker: stage timings via \(wrapped.joined(separator: "; "))\n"
                        .utf8)
            )
        }
        try writeResponse(.ready(loadSeconds: loadSeconds))

        let input = FileHandle.standardInput
        let converter = AudioConverter()
        while let header = try readExactly(requestHeaderBytes, from: input) {
            do {
                let request = try parseHeader(header)
                let payloadBytes = request.sampleCount * MemoryLayout<Float>.size
                guard let payload = try readExactly(payloadBytes, from: input) else {
                    throw WorkerError.truncatedRequest
                }
                let samples = decodeSamples(payload, count: request.sampleCount)

                let resampleStart = ContinuousClock.now
                let modelSamples = try converter.resample(samples, from: Double(request.sampleRate))
                let resampleSeconds = seconds(since: resampleStart)

                let thresholdSamples = Int(options.longRegimeSeconds) * 16_000
                let isLongRegime = modelSamples.count >= thresholdSamples
                // Buckets are loaded with the short regime's compute units, so
                // they may only serve a request the regime would have run with
                // those same units.
                let regimeComputeUnits =
                    isLongRegime ? options.longComputeUnits : options.shortComputeUnits
                let bucketManager =
                    regimeComputeUnits == options.shortComputeUnits
                    ? buckets.manager(forSampleCount: modelSamples.count) : nil
                let manager =
                    bucketManager
                    ?? (isLongRegime ? (longManager ?? shortManager) : shortManager)
                let decodeStart = ContinuousClock.now
                let profileStart =
                    options.emitStageTimings ? StageProfiler.shared.beginUtterance() : 0
                // `transcribeWithTimings` runs the same decode as
                // `transcribe` and reads the emission frames the greedy RNNT
                // decoder already recorded, so the token spans cost nothing
                // beyond the frame→seconds conversion.
                let transcription = try await manager.transcribeWithTimings(modelSamples)
                let text = transcription.text
                var stages =
                    options.emitStageTimings
                    ? StageProfiler.shared.endUtterance(
                        startNanoseconds: profileStart, endNanoseconds: StageProfiler.now()
                    )
                    : nil
                stages?.resampleMs = resampleSeconds * 1_000
                let decodeSeconds = seconds(since: decodeStart)
                try writeResponse(
                    .result(
                        text: text,
                        decodeSeconds: decodeSeconds,
                        resampleSeconds: resampleSeconds,
                        stages: stages,
                        tokenSpans: transcription.tokenTimings.map {
                            TokenSpan(text: $0.token, startS: $0.startTime, endS: $0.endTime)
                        }
                    )
                )
            } catch {
                try writeResponse(.failure(kind: "result", error: error))
            }
        }
    }

    /// The native decode loop, or nil to keep FluidAudio's CoreML one.
    ///
    /// A failure to build it is thrown rather than swallowed: it means the
    /// compiled decoder or joint is not the program this reimplements, and
    /// silently falling back would hide a model swap behind a latency
    /// regression.
    private static func rnntDecoderFactory(
        _ options: WorkerOptions
    ) -> UnifiedAsrManager.UnifiedRnntDecoderFactory? {
        guard options.nativeRnnt else { return nil }
        return { directory, config in
            try NativeRnntDecoder(modelDirectory: directory, config: config)
        }
    }

    private static func loadManager(
        computeUnits: MLComputeUnits,
        options: WorkerOptions
    ) async throws -> UnifiedAsrManager {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let manager = UnifiedAsrManager(
            configuration: configuration,
            encoderPrecision: .int8,
            rnntDecoderFactory: rnntDecoderFactory(options)
        )
        if let modelDirectory = options.modelDirectory {
            try await manager.loadModels(from: modelDirectory)
        } else {
            try await manager.loadModels(to: options.modelRoot, configuration: nil)
        }
        return manager
    }
}

private struct RequestHeader {
    let sampleRate: UInt32
    let sampleCount: Int
}

private func parseHeader(_ data: Data) throws -> RequestHeader {
    guard data.prefix(protocolMagic.count) == protocolMagic else {
        throw WorkerError.invalidMagic
    }
    let version = readUInt32(data, at: 4)
    guard version == protocolVersion else {
        throw WorkerError.unsupportedVersion(version)
    }
    let sampleRate = readUInt32(data, at: 8)
    guard (8_000...384_000).contains(sampleRate) else {
        throw WorkerError.invalidSampleRate(sampleRate)
    }
    let sampleCount = readUInt32(data, at: 12)
    let maximumSampleCount = UInt64(sampleRate) * maximumAudioSeconds
    guard sampleCount > 0, UInt64(sampleCount) <= maximumSampleCount else {
        throw WorkerError.invalidSampleCount(sampleCount)
    }
    return RequestHeader(sampleRate: sampleRate, sampleCount: Int(sampleCount))
}

private func readExactly(_ count: Int, from handle: FileHandle) throws -> Data? {
    var result = Data()
    result.reserveCapacity(count)
    while result.count < count {
        let chunk = try handle.read(upToCount: count - result.count) ?? Data()
        if chunk.isEmpty {
            if result.isEmpty { return nil }
            throw WorkerError.truncatedRequest
        }
        result.append(chunk)
    }
    return result
}

private func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
    UInt32(data[offset])
        | UInt32(data[offset + 1]) << 8
        | UInt32(data[offset + 2]) << 16
        | UInt32(data[offset + 3]) << 24
}

private func decodeSamples(_ data: Data, count: Int) -> [Float] {
    var samples = [Float](repeating: 0, count: count)
    _ = samples.withUnsafeMutableBytes { destination in
        data.copyBytes(to: destination)
    }
    return samples
}

private func writeResponse(_ response: WorkerResponse) throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.sortedKeys]
    let payload = try encoder.encode(response)
    guard payload.count <= maximumResponseBytes else {
        throw WorkerError.oversizedResponse(payload.count)
    }
    var frame = Data()
    frame.reserveCapacity(MemoryLayout<UInt32>.size + payload.count)
    appendUInt32(UInt32(payload.count), to: &frame)
    frame.append(payload)
    try FileHandle.standardOutput.write(contentsOf: frame)
}

private func appendUInt32(_ value: UInt32, to data: inout Data) {
    data.append(UInt8(truncatingIfNeeded: value))
    data.append(UInt8(truncatingIfNeeded: value >> 8))
    data.append(UInt8(truncatingIfNeeded: value >> 16))
    data.append(UInt8(truncatingIfNeeded: value >> 24))
}

private func seconds(since start: ContinuousClock.Instant) -> Double {
    let duration = start.duration(to: .now)
    return Double(duration.components.seconds)
        + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
}
