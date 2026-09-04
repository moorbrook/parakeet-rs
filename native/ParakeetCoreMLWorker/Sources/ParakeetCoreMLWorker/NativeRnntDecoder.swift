import Accelerate
import CoreML
import FluidAudio
import Foundation

/// The greedy RNNT decode loop, run natively instead of through Core ML.
///
/// FluidAudio's `UnifiedRnntDecoder` issues one `decoderModel.prediction` per
/// emitted token and one `jointDecisionModel.prediction` per frame plus one per
/// token. Each is a separate Core ML call with a dispatch floor around 100 µs,
/// which at 4.85 s of audio is 132 calls and the largest remaining stage of the
/// worker. Neither model has a Neural Engine path — the prediction network is a
/// two-layer LSTM, which Core ML runs `cpuOnly` by necessity
/// (`docs/asr/COMPUTE_PLAN.md`) — so the dispatch buys nothing but the driver
/// round trip.
///
/// This runs the same two programs directly: the embedding gather and LSTM from
/// `parakeet_unified_decoder.mlmodelc`, the three projections and the argmax
/// from `parakeet_unified_joint_decision_single_step.mlmodelc`, on the weights
/// read out of those same bundles. The loop, the blank handling and the
/// symbol-per-frame cap are FluidAudio's, reproduced step for step.
///
/// ## How closely it matches
///
/// Not bit-exactly, and it cannot: Core ML's CPU `lstm` accumulates its
/// recurrent product in fp16, while this accumulates in fp32 over the same fp16
/// weights. `scripts/capture-rnnt-parity.py` measures the gap — the decoder
/// output the joint consumes agrees to 0.014 over a 32-step trajectory against
/// cell values reaching 9.2, and the divergence does not compound. The fp32
/// accumulation is the more accurate of the two. Every fp16 rounding the
/// exported program performs at an operation boundary is reproduced, since
/// those are what the model was trained through rather than an artifact of the
/// kernel.
final class NativeRnntDecoder {
    /// Row blocking of the LSTM gate matrices, decided by measurement against
    /// the compiled model rather than by reading the MIL: input, forget,
    /// output, cell. See `scripts/capture-rnnt-parity.py`.
    private enum Gate: Int {
        case input = 0
        case forget = 1
        case output = 2
        case cell = 3
    }

    private let prediction: RnntPredictionNetwork
    private let joint: RnntJointNetwork
    private let blankIndex: Int
    private let maximumSymbolsPerFrame: Int

    private var hiddenState: [Float]
    private var cellState: [Float]
    private var lastToken: Int

    /// Shared, immutable weights. Bucketed encoders mean several
    /// `UnifiedAsrManager` instances over one model directory, and each would
    /// otherwise hold its own 15 MB copy.
    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var cache: [String: (RnntPredictionNetwork, RnntJointNetwork)] = [:]

    /// Row slices the matrix kernels split across. Overridable because the best
    /// value is a property of the machine, not of the model.
    static let concurrency: Int = {
        if let raw = ProcessInfo.processInfo.environment["PARAKEET_RNNT_THREADS"],
            let value = Int(raw), value >= 1
        {
            return value
        }
        return 4
    }()

    init(modelDirectory: URL, config: UnifiedConfig) throws {
        let key = modelDirectory.standardizedFileURL.path
        Self.cacheLock.lock()
        let cached = Self.cache[key]
        Self.cacheLock.unlock()
        if let cached {
            (prediction, joint) = cached
        } else {
            let loaded = (
                try RnntPredictionNetwork(
                    bundle: modelDirectory.appendingPathComponent(
                        ModelNames.ParakeetUnified.decoderFile),
                    layers: config.decoderLayers,
                    hidden: config.decoderHidden
                ),
                try RnntJointNetwork(
                    bundle: modelDirectory.appendingPathComponent(
                        ModelNames.ParakeetUnified.jointDecisionFile))
            )
            Self.cacheLock.lock()
            Self.cache[key] = loaded
            Self.cacheLock.unlock()
            (prediction, joint) = loaded
        }
        guard prediction.vocabulary == joint.vocabulary,
            prediction.hidden == joint.decoderDimension
        else {
            throw MilWeightBlob.LoadError.shapeMismatch(
                "prediction network against joint",
                [prediction.vocabulary, prediction.hidden],
                [joint.vocabulary, joint.decoderDimension]
            )
        }
        blankIndex = config.blankIdx
        maximumSymbolsPerFrame = config.maxSymbolsPerFrame
        hiddenState = [Float](repeating: 0, count: config.decoderLayers * config.decoderHidden)
        cellState = hiddenState
        lastToken = config.blankIdx
    }

    /// The encoder width this decoder was built for, so a mismatched encoder
    /// output is caught before it is read as one.
    var encoderDimension: Int { joint.encoderDimension }
}

extension NativeRnntDecoder: UnifiedRnntDecoding {
    func reset() throws {
        for index in hiddenState.indices {
            hiddenState[index] = 0
            cellState[index] = 0
        }
        lastToken = blankIndex
    }

    func decode(
        encoded: MLMultiArray,
        frameRange: Range<Int>,
        globalFrameOffset: Int
    ) throws -> [UnifiedRnntEmission] {
        let projected = try joint.projectEncoder(encoded, frameRange: frameRange)

        var currentToken = lastToken
        var currentHidden = hiddenState
        var currentCell = cellState
        var emissions: [UnifiedRnntEmission] = []

        var step = predictionStep(token: currentToken, hidden: currentHidden, cell: currentCell)
        var decoderProjection = jointProjectDecoder(step.output)

        for frame in frameRange {
            let encoderProjection = projected.frame(frame - frameRange.lowerBound)
            for _ in 0..<maximumSymbolsPerFrame {
                let decision = jointDecide(
                    encoderProjection: encoderProjection, decoderProjection: decoderProjection)
                if decision.token == blankIndex { break }
                emissions.append(
                    UnifiedRnntEmission(
                        token: decision.token,
                        frame: globalFrameOffset + frame,
                        prob: decision.probability
                    )
                )
                currentToken = decision.token
                currentHidden = step.hidden
                currentCell = step.cell
                step = predictionStep(
                    token: currentToken, hidden: currentHidden, cell: currentCell)
                decoderProjection = jointProjectDecoder(step.output)
            }
        }

        lastToken = currentToken
        hiddenState = currentHidden
        cellState = currentCell
        return emissions
    }

    /// One prediction-network step, timed as the native counterpart of a
    /// `decoderModel.prediction` dispatch.
    private func predictionStep(
        token: Int, hidden: [Float], cell: [Float]
    ) -> RnntPredictionNetwork.Step {
        let start = StageProfiler.shared.nativeStart()
        let step = prediction.step(token: token, hidden: hidden, cell: cell)
        StageProfiler.shared.recordNative(.nativeDecoder, since: start)
        return step
    }

    /// The joint's decoder-side projection. It changes only when a token is
    /// emitted, so it is hoisted out of the symbol loop the way the encoder
    /// projection is hoisted out of the frame loop; the compiled model
    /// recomputed both on every call.
    private func jointProjectDecoder(_ decoderOutput: [Float]) -> [Float] {
        let start = StageProfiler.shared.nativeStart()
        let projection = joint.projectDecoder(decoderOutput)
        StageProfiler.shared.recordNative(.nativeJoint, since: start)
        return projection
    }

    private func jointDecide(
        encoderProjection: UnsafePointer<Float>, decoderProjection: [Float]
    ) -> RnntJointNetwork.Decision {
        let start = StageProfiler.shared.nativeStart()
        let decision = joint.decide(
            encoderProjection: encoderProjection,
            decoderProjection: decoderProjection,
            blankIndex: blankIndex
        )
        StageProfiler.shared.recordNative(.nativeJoint, since: start)
        return decision
    }
}

// MARK: - Prediction network

/// The embedding gather and two-layer LSTM of `parakeet_unified_decoder`.
final class RnntPredictionNetwork {
    struct Step {
        let output: [Float]
        let hidden: [Float]
        let cell: [Float]
    }

    let layers: Int
    let hidden: Int
    let vocabulary: Int

    private let embedding: [Float]
    /// Per layer, `W_ih` and `W_hh` interleaved row by row: `[4H, 2H]`.
    private let gates: [Float16Matrix]
    private let biases: [[Float]]

    init(bundle: URL, layers: Int, hidden: Int) throws {
        let blob = try MilWeightBlob(bundle: bundle)
        self.layers = layers
        self.hidden = hidden

        // The embedding is the only tensor whose leading dimension is not
        // implied by the config, so the vocabulary size is read off it.
        let embeddingShape = try blob.shape("module_prediction_embed_weight_to_fp16")
        guard embeddingShape.count == 2, embeddingShape[1] == hidden else {
            throw MilWeightBlob.LoadError.shapeMismatch(
                "module_prediction_embed_weight_to_fp16", embeddingShape, [-1, hidden])
        }
        vocabulary = embeddingShape[0]
        embedding = try blob.fp32(
            "module_prediction_embed_weight_to_fp16", shape: embeddingShape)

        // MIL names the two LSTM layers' constants by concatenation order:
        // layer 0 takes bias 0, weight_ih 1, weight_hh 2; layer 1 takes 3, 4, 5.
        var gates: [Float16Matrix] = []
        var biases: [[Float]] = []
        for (biasIndex, inputIndex, hiddenIndex) in [(0, 1, 2), (3, 4, 5)].prefix(layers) {
            let inputWeights = try blob.fp16(
                "concat_\(inputIndex)_to_fp16", shape: [4 * hidden, hidden])
            let hiddenWeights = try blob.fp16(
                "concat_\(hiddenIndex)_to_fp16", shape: [4 * hidden, hidden])
            gates.append(
                Float16Matrix(
                    interleavingRowsOf: inputWeights, hiddenWeights, rows: 4 * hidden))
            biases.append(
                try blob.fp32("concat_\(biasIndex)_to_fp16", shape: [4 * hidden]))
        }
        self.gates = gates
        self.biases = biases
    }

    /// One `(token, h, c)` step. `hidden` and `cell` are `layers * hidden` long.
    func step(token: Int, hidden state: [Float], cell: [Float]) -> Step {
        let size = hidden
        var input = [Float](repeating: 0, count: 2 * size)
        var gateValues = [Float](repeating: 0, count: 4 * size)
        var scratch = [Float](repeating: 0, count: 3 * size)
        var newHidden = state
        var newCell = cell

        embedding.withUnsafeBufferPointer { source in
            input.withUnsafeMutableBufferPointer { destination in
                destination.baseAddress!.update(
                    from: source.baseAddress! + token * size, count: size)
            }
        }

        for layer in 0..<layers {
            let offset = layer * size
            state.withUnsafeBufferPointer { source in
                input.withUnsafeMutableBufferPointer { vector in
                    vector.baseAddress!.advanced(by: size).update(
                        from: source.baseAddress! + offset, count: size)
                }
            }
            input.withUnsafeBufferPointer { vector in
                biases[layer].withUnsafeBufferPointer { bias in
                    gateValues.withUnsafeMutableBufferPointer { out in
                        gates[layer].multiply(
                            vector: vector.baseAddress!,
                            bias: bias.baseAddress!,
                            into: out.baseAddress!,
                            chunks: NativeRnntDecoder.concurrency
                        )
                    }
                }
            }
            Self.applyGates(
                gateValues: &gateValues,
                scratch: &scratch,
                size: size,
                cell: &newCell,
                hidden: &newHidden,
                offset: offset
            )
            input.withUnsafeMutableBufferPointer { vector in
                newHidden.withUnsafeBufferPointer { source in
                    vector.baseAddress!.update(from: source.baseAddress! + offset, count: size)
                }
            }
        }

        let output = Array(newHidden[((layers - 1) * size)..<(layers * size)])
        return Step(output: output, hidden: newHidden, cell: newCell)
    }

    /// `i, f, o = sigmoid(gates)`, `g = tanh(gates)`, then the cell and hidden
    /// updates, each rounded to fp16 as the compiled program's tensors are.
    private static func applyGates(
        gateValues: inout [Float],
        scratch: inout [Float],
        size: Int,
        cell: inout [Float],
        hidden: inout [Float],
        offset: Int
    ) {
        var count = Int32(3 * size)
        gateValues.withUnsafeMutableBufferPointer { gates in
            scratch.withUnsafeMutableBufferPointer { temporary in
                // sigmoid(x) = 1 / (1 + exp(-x)), over the input, forget and
                // output blocks in one vForce call.
                var negativeOne: Float = -1
                vDSP_vsmul(gates.baseAddress!, 1, &negativeOne, temporary.baseAddress!, 1,
                    vDSP_Length(3 * size))
                vvexpf(temporary.baseAddress!, temporary.baseAddress!, &count)
                var one: Float = 1
                vDSP_vsadd(temporary.baseAddress!, 1, &one, temporary.baseAddress!, 1,
                    vDSP_Length(3 * size))
                vvrecf(gates.baseAddress!, temporary.baseAddress!, &count)
                var cellCount = Int32(size)
                vvtanhf(
                    gates.baseAddress! + 3 * size, gates.baseAddress! + 3 * size, &cellCount)
            }
        }
        gateValues.withUnsafeBufferPointer { gates in
            let inputGate = gates.baseAddress!
            let forgetGate = inputGate + size
            let outputGate = inputGate + 2 * size
            let cellGate = inputGate + 3 * size
            cell.withUnsafeMutableBufferPointer { cellState in
                hidden.withUnsafeMutableBufferPointer { hiddenState in
                    scratch.withUnsafeMutableBufferPointer { temporary in
                        let target = cellState.baseAddress! + offset
                        // c' = f * c + i * g
                        vDSP_vmul(forgetGate, 1, target, 1, temporary.baseAddress!, 1,
                            vDSP_Length(size))
                        vDSP_vma(inputGate, 1, cellGate, 1, temporary.baseAddress!, 1, target, 1,
                            vDSP_Length(size))
                        roundToFloat16(target, count: size)
                        var tanhCount = Int32(size)
                        vvtanhf(temporary.baseAddress!, target, &tanhCount)
                        // h' = o * tanh(c')
                        vDSP_vmul(outputGate, 1, temporary.baseAddress!, 1,
                            hiddenState.baseAddress! + offset, 1, vDSP_Length(size))
                        roundToFloat16(hiddenState.baseAddress! + offset, count: size)
                    }
                }
            }
        }
    }
}

// MARK: - Joint decision

/// The three projections and argmax of
/// `parakeet_unified_joint_decision_single_step`.
final class RnntJointNetwork {
    struct Decision {
        let token: Int
        let probability: Float
    }

    /// Encoder projections for one window, `frames` rows of `stride` values.
    ///
    /// Owns its storage rather than wrapping an array so a row pointer stays
    /// valid for the length of the decode, which is what lets the frame loop
    /// hand one row straight to the joint without copying it.
    final class ProjectedFrames {
        let frames: Int
        let stride: Int
        private let storage: UnsafeMutablePointer<Float>

        init(frames: Int, stride: Int) {
            self.frames = frames
            self.stride = stride
            storage = UnsafeMutablePointer<Float>.allocate(capacity: max(frames * stride, 1))
            storage.initialize(repeating: 0, count: max(frames * stride, 1))
        }

        deinit {
            storage.deinitialize(count: max(frames * stride, 1))
            storage.deallocate()
        }

        var buffer: UnsafeMutablePointer<Float> { storage }

        func frame(_ index: Int) -> UnsafePointer<Float> {
            UnsafePointer(storage + index * stride)
        }
    }

    let encoderDimension: Int
    let decoderDimension: Int
    let vocabulary: Int

    /// The encoder projection is the one matrix applied to every frame of a
    /// window rather than once per step, so it is held as fp32 and run through
    /// BLAS as a single matrix product instead of a matrix-vector product per
    /// frame.
    private let encoderWeights: [Float]
    private let encoderBias: [Float]
    private let decoderProjection: Float16Matrix
    private let decoderBias: [Float]
    private let outputProjection: Float16Matrix
    private let outputBias: [Float]

    init(bundle: URL) throws {
        let blob = try MilWeightBlob(bundle: bundle)
        let encoderShape = try blob.shape("joint_module_enc_weight_to_fp16")
        let outputShape = try blob.shape("joint_module_joint_net_2_weight_to_fp16")
        guard encoderShape.count == 2, outputShape.count == 2, encoderShape[0] == outputShape[1]
        else {
            throw MilWeightBlob.LoadError.shapeMismatch(
                "joint projections", encoderShape + outputShape, [])
        }
        decoderDimension = encoderShape[0]
        encoderDimension = encoderShape[1]
        vocabulary = outputShape[0]
        encoderWeights = try blob.fp32(
            "joint_module_enc_weight_to_fp16", shape: [decoderDimension, encoderDimension])
        encoderBias = try blob.fp32("joint_module_enc_bias_to_fp16", shape: [decoderDimension])
        decoderProjection = Float16Matrix(
            rows: decoderDimension,
            columns: decoderDimension,
            values: try blob.fp16(
                "joint_module_pred_weight_to_fp16",
                shape: [decoderDimension, decoderDimension])
        )
        decoderBias = try blob.fp32("joint_module_pred_bias_to_fp16", shape: [decoderDimension])
        outputProjection = Float16Matrix(
            rows: vocabulary,
            columns: decoderDimension,
            values: try blob.fp16(
                "joint_module_joint_net_2_weight_to_fp16", shape: [vocabulary, decoderDimension])
        )
        outputBias = try blob.fp32("joint_module_joint_net_2_bias_to_fp16", shape: [vocabulary])
    }

    enum ProjectionError: LocalizedError {
        case unexpectedEncoderShape([Int])

        var errorDescription: String? {
            switch self {
            case .unexpectedEncoderShape(let shape):
                "encoder output has shape \(shape), which this joint cannot project"
            }
        }
    }

    /// Project every frame of the window at once.
    ///
    /// The compiled joint reprojects the encoder frame on every call, which the
    /// greedy loop makes once per frame plus once per emitted token. The
    /// projection depends only on the frame, so the whole window is one matrix
    /// product here — and a matrix product is where BLAS is at its best, unlike
    /// the matrix-vector products the rest of the loop is made of.
    func projectEncoder(_ encoded: MLMultiArray, frameRange: Range<Int>) throws -> ProjectedFrames {
        let shape = encoded.shape.map(\.intValue)
        guard shape.count == 3, shape[1] == encoderDimension, encoded.dataType == .float32 else {
            throw ProjectionError.unexpectedEncoderShape(shape)
        }
        let frames = frameRange.count
        let projected = ProjectedFrames(frames: frames, stride: decoderDimension)
        guard frames > 0 else { return projected }

        let channelStride = encoded.strides[1].intValue
        let timeStride = encoded.strides[2].intValue
        var packed = [Float](repeating: 0, count: encoderDimension * frames)
        encoded.withUnsafeBufferPointer(ofType: Float.self) { source in
            let base = source.baseAddress!
            packed.withUnsafeMutableBufferPointer { destination in
                for channel in 0..<encoderDimension {
                    let row = destination.baseAddress! + channel * frames
                    let column = base + channel * channelStride
                    if timeStride == 1 {
                        row.update(from: column + frameRange.lowerBound, count: frames)
                    } else {
                        for index in 0..<frames {
                            row[index] = column[(frameRange.lowerBound + index) * timeStride]
                        }
                    }
                }
            }
        }
        // The compiled program casts `encoder_step` to fp16 on entry.
        packed.withUnsafeMutableBufferPointer {
            roundToFloat16($0.baseAddress!, count: $0.count)
        }

        packed.withUnsafeBufferPointer { input in
            encoderWeights.withUnsafeBufferPointer { weights in
                // [frames, D] = [frames, E] x [E, D], from [E, frames] and
                // [D, E] as stored.
                cblas_sgemm(
                    CblasRowMajor, CblasTrans, CblasTrans,
                    Int32(frames), Int32(decoderDimension), Int32(encoderDimension),
                    1, input.baseAddress!, Int32(frames),
                    weights.baseAddress!, Int32(encoderDimension),
                    0, projected.buffer, Int32(decoderDimension))
            }
        }
        encoderBias.withUnsafeBufferPointer { bias in
            for frame in 0..<frames {
                let row = projected.buffer + frame * decoderDimension
                vDSP_vadd(row, 1, bias.baseAddress!, 1, row, 1, vDSP_Length(decoderDimension))
                roundToFloat16(row, count: decoderDimension)
            }
        }
        return projected
    }

    func projectDecoder(_ decoderOutput: [Float]) -> [Float] {
        var projected = [Float](repeating: 0, count: decoderDimension)
        decoderOutput.withUnsafeBufferPointer { vector in
            decoderBias.withUnsafeBufferPointer { bias in
                projected.withUnsafeMutableBufferPointer { output in
                    decoderProjection.multiply(
                        vector: vector.baseAddress!,
                        bias: bias.baseAddress!,
                        into: output.baseAddress!,
                        chunks: NativeRnntDecoder.concurrency
                    )
                }
            }
        }
        return projected
    }

    /// `argmax(W · relu(enc + dec) + b)`, plus the softmax probability of the
    /// winner when it is not blank.
    ///
    /// The probability is what FluidAudio records as per-token confidence, and
    /// the loop discards it on a blank, so it is computed on emission only. The
    /// compiled model computes it every call because a Core ML program has no
    /// way to skip an operation.
    func decide(
        encoderProjection: UnsafePointer<Float>,
        decoderProjection: [Float],
        blankIndex: Int
    ) -> Decision {
        var activated = [Float](repeating: 0, count: decoderDimension)
        decoderProjection.withUnsafeBufferPointer { decoder in
            activated.withUnsafeMutableBufferPointer { output in
                vDSP_vadd(
                    encoderProjection, 1, decoder.baseAddress!, 1, output.baseAddress!, 1,
                    vDSP_Length(decoderDimension))
                roundToFloat16(output.baseAddress!, count: decoderDimension)
                var zero: Float = 0
                vDSP_vthres(
                    output.baseAddress!, 1, &zero, output.baseAddress!, 1,
                    vDSP_Length(decoderDimension))
            }
        }

        var logits = [Float](repeating: 0, count: vocabulary)
        activated.withUnsafeBufferPointer { vector in
            outputBias.withUnsafeBufferPointer { bias in
                logits.withUnsafeMutableBufferPointer { output in
                    outputProjection.multiply(
                        vector: vector.baseAddress!,
                        bias: bias.baseAddress!,
                        into: output.baseAddress!,
                        chunks: NativeRnntDecoder.concurrency
                    )
                }
            }
        }

        var best: Float = 0
        var index: vDSP_Length = 0
        logits.withUnsafeBufferPointer { values in
            vDSP_maxvi(values.baseAddress!, 1, &best, &index, vDSP_Length(vocabulary))
        }
        let token = Int(index)
        guard token != blankIndex else { return Decision(token: token, probability: 0) }

        var shifted = logits
        var negatedBest = -best
        var total: Float = 0
        shifted.withUnsafeMutableBufferPointer { values in
            vDSP_vsadd(values.baseAddress!, 1, &negatedBest, values.baseAddress!, 1,
                vDSP_Length(vocabulary))
            var count = Int32(vocabulary)
            vvexpf(values.baseAddress!, values.baseAddress!, &count)
            vDSP_sve(values.baseAddress!, 1, &total, vDSP_Length(vocabulary))
        }
        return Decision(token: token, probability: Float(Float16(shifted[token] / total)))
    }
}
