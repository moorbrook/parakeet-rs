import CoreML
import FluidAudio
import Foundation

/// FluidInference's 32 ms Silero graph uses the same explicit recurrent state
/// as FluidAudio. VadManager pads to its fixed 256 ms graph, so this adapter
/// supplies the 512-sample graph directly to keep capture's frame cadence.
final class CoreMlVad {
    private let model: MLModel
    private var state = VadState.initial()

    init(modelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    func probability(_ samples: [Float]) throws -> Float {
        guard samples.count == 512 else {
            throw WorkerError.invalidArgument("VAD requires a 512-sample frame")
        }
        return try autoreleasepool {
            let audio = try MLMultiArray(shape: [1, 576], dataType: .float32)
            let hidden = try MLMultiArray(shape: [1, 128], dataType: .float32)
            let cell = try MLMultiArray(shape: [1, 128], dataType: .float32)
            for (index, sample) in (state.context + samples).enumerated() {
                audio[index] = NSNumber(value: sample)
            }
            for index in 0..<128 {
                hidden[index] = NSNumber(value: state.hiddenState[index])
                cell[index] = NSNumber(value: state.cellState[index])
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "audio_input": audio, "hidden_state": hidden, "cell_state": cell,
            ])
            let result = try model.prediction(from: input)
            guard let probability = result.featureValue(for: "vad_output")?.multiArrayValue,
                let nextHidden = result.featureValue(for: "new_hidden_state")?.multiArrayValue,
                let nextCell = result.featureValue(for: "new_cell_state")?.multiArrayValue,
                probability.count == 1, nextHidden.count == 128, nextCell.count == 128
            else {
                throw WorkerError.invalidArgument("Core ML VAD output shape does not match its pinned contract")
            }
            state = VadState(
                hiddenState: (0..<128).map { nextHidden[$0].floatValue },
                cellState: (0..<128).map { nextCell[$0].floatValue },
                context: Array(samples.suffix(VadState.contextLength))
            )
            return probability[0].floatValue
        }
    }
}
