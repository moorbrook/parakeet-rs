// Encoder shape probe: loads a compiled encoder, prints its declared input and
// output shapes, and times a few predictions on zero-filled inputs.
//
// It answers one question the stage profiler cannot: how encoder cost varies
// with the compiled mel length, which decides whether bucketed fixed shapes are
// worth building. It is a diagnostic and links no FluidAudio code.
import CoreML
import Foundation

private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("parakeet-encoder-probe: \(message)\n".utf8))
    exit(1)
}

private struct Options {
    var modelURL: URL?
    var computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    var warmups = 3
    var repetitions = 10
}

private func parseOptions() -> Options {
    var options = Options()
    let arguments = Array(CommandLine.arguments.dropFirst())
    var index = 0
    while index < arguments.count {
        switch arguments[index] {
        case "--compute-units":
            index += 1
            guard index < arguments.count else { fail("--compute-units needs a name") }
            switch arguments[index] {
            case "all": options.computeUnits = .all
            case "cpu-and-gpu": options.computeUnits = .cpuAndGPU
            case "cpu-and-neural-engine": options.computeUnits = .cpuAndNeuralEngine
            case "cpu-only": options.computeUnits = .cpuOnly
            default: fail("unknown compute units \(arguments[index])")
            }
        case "--warmups":
            index += 1
            guard index < arguments.count, let value = Int(arguments[index]), value >= 0 else {
                fail("--warmups needs a non-negative integer")
            }
            options.warmups = value
        case "--reps":
            index += 1
            guard index < arguments.count, let value = Int(arguments[index]), value > 0 else {
                fail("--reps needs a positive integer")
            }
            options.repetitions = value
        case "-h", "--help":
            FileHandle.standardError.write(
                Data(
                    "usage: parakeet-encoder-probe [--compute-units NAME] [--warmups N] [--reps N] MODEL.mlmodelc\n"
                        .utf8))
            exit(0)
        default:
            guard options.modelURL == nil else { fail("only one model path is accepted") }
            options.modelURL = URL(fileURLWithPath: arguments[index], isDirectory: true)
        }
        index += 1
    }
    guard options.modelURL != nil else { fail("no model path given") }
    return options
}

private final class ZeroFeatureProvider: NSObject, MLFeatureProvider {
    private let values: [String: MLFeatureValue]
    let featureNames: Set<String>

    init(description: MLModelDescription, melLength: Int?) throws {
        var values: [String: MLFeatureValue] = [:]
        for (name, input) in description.inputDescriptionsByName {
            guard let constraint = input.multiArrayConstraint else {
                throw NSError(
                    domain: "probe", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "input \(name) is not a multi-array"])
            }
            let array = try MLMultiArray(
                shape: constraint.shape, dataType: constraint.dataType)
            // A length input must name the real frame count, or the encoder's
            // mask chain would treat the whole window as padding.
            if constraint.shape.count == 1, constraint.shape[0].intValue == 1,
                let melLength
            {
                array[0] = NSNumber(value: melLength)
            }
            values[name] = MLFeatureValue(multiArray: array)
        }
        self.values = values
        self.featureNames = Set(values.keys)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? { values[featureName] }
}

private func shape(_ constraint: MLMultiArrayConstraint?) -> String {
    guard let constraint else { return "?" }
    return "[" + constraint.shape.map { $0.stringValue }.joined(separator: ", ") + "]"
}

@available(macOS 14.4, *)
private func run() async {
    let options = parseOptions()
    guard let url = options.modelURL else { fail("no model path given") }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = options.computeUnits

    let loadStart = ContinuousClock.now
    let model: MLModel
    do {
        model = try await MLModel.load(contentsOf: url, configuration: configuration)
    } catch {
        fail("could not load \(url.lastPathComponent): \(error)")
    }
    let loadMs = Double(loadStart.duration(to: .now).components.attoseconds) / 1e15
        + Double(loadStart.duration(to: .now).components.seconds) * 1000

    let description = model.modelDescription
    print("model: \(url.lastPathComponent)")
    print("compute units: \(options.computeUnits.rawValue)")
    print("load: \(String(format: "%.1f", loadMs)) ms")
    for name in description.inputDescriptionsByName.keys.sorted() {
        print("  input  \(name) \(shape(description.inputDescriptionsByName[name]?.multiArrayConstraint))")
    }
    for name in description.outputDescriptionsByName.keys.sorted() {
        print("  output \(name) \(shape(description.outputDescriptionsByName[name]?.multiArrayConstraint))")
    }

    // The mel input is the only rank-3 input; its last extent is the frame count.
    let melFrames =
        description.inputDescriptionsByName.values
        .compactMap { $0.multiArrayConstraint }
        .filter { $0.shape.count == 3 }
        .map { $0.shape[2].intValue }
        .first
    print("mel frames: \(melFrames.map(String.init) ?? "unknown")")

    let provider: ZeroFeatureProvider
    do {
        provider = try ZeroFeatureProvider(description: description, melLength: melFrames)
    } catch {
        fail("could not build inputs: \(error)")
    }

    var samples: [Double] = []
    for iteration in 0..<(options.warmups + options.repetitions) {
        let start = ContinuousClock.now
        do {
            _ = try await model.prediction(from: provider)
        } catch {
            fail("prediction failed: \(error)")
        }
        let duration = start.duration(to: .now)
        let ms = Double(duration.components.seconds) * 1000
            + Double(duration.components.attoseconds) / 1e15
        if iteration >= options.warmups { samples.append(ms) }
    }
    samples.sort()
    let median = samples[samples.count / 2]
    print(
        "predict n=\(samples.count) min=\(String(format: "%.2f", samples[0])) "
            + "p50=\(String(format: "%.2f", median)) "
            + "max=\(String(format: "%.2f", samples[samples.count - 1])) ms"
    )
}

if #available(macOS 14.4, *) {
    await run()
} else {
    fail("this probe requires macOS 14.4 or newer")
}
