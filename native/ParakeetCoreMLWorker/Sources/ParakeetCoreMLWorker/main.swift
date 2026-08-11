import CoreML
import FluidAudio
import Foundation

private let protocolMagic = Data([0x50, 0x52, 0x4b, 0x54]) // "PRKT"
private let protocolVersion: UInt32 = 1
private let requestHeaderBytes = 16
private let maximumAudioSeconds: UInt64 = 30 * 60
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

    static func parse(_ arguments: [String]) throws -> Self {
        var modelDirectory: URL?
        var modelRoot: URL?
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
            case "-h", "--help":
                FileHandle.standardError.write(
                    Data(
                        "usage: parakeet-coreml-worker [--model-dir DIR | --model-root DIR]\n"
                            .utf8
                    )
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
        return Self(modelDirectory: modelDirectory, modelRoot: modelRoot)
    }
}

private struct WorkerResponse: Encodable {
    let kind: String
    let ok: Bool
    let text: String?
    let error: String?
    let loadSeconds: Double?
    let decodeSeconds: Double?
    let resampleSeconds: Double?

    static func ready(loadSeconds: Double) -> Self {
        Self(
            kind: "ready",
            ok: true,
            text: nil,
            error: nil,
            loadSeconds: loadSeconds,
            decodeSeconds: nil,
            resampleSeconds: nil
        )
    }

    static func result(text: String, decodeSeconds: Double, resampleSeconds: Double) -> Self {
        Self(
            kind: "result",
            ok: true,
            text: text,
            error: nil,
            loadSeconds: nil,
            decodeSeconds: decodeSeconds,
            resampleSeconds: resampleSeconds
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
            resampleSeconds: nil
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
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let manager = UnifiedAsrManager(
            configuration: configuration,
            encoderPrecision: .int8
        )

        let loadStart = ContinuousClock.now
        if let modelDirectory = options.modelDirectory {
            try await manager.loadModels(from: modelDirectory)
        } else {
            try await manager.loadModels(to: options.modelRoot, configuration: nil)
        }
        let loadSeconds = seconds(since: loadStart)
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

                let decodeStart = ContinuousClock.now
                let text = try await manager.transcribe(modelSamples)
                let decodeSeconds = seconds(since: decodeStart)
                try writeResponse(
                    .result(
                        text: text,
                        decodeSeconds: decodeSeconds,
                        resampleSeconds: resampleSeconds
                    )
                )
            } catch {
                try writeResponse(.failure(kind: "result", error: error))
            }
        }
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
