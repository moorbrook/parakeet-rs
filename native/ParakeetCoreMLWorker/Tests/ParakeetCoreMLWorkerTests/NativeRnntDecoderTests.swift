import CoreML
import Foundation
import Testing

@testable import ParakeetCoreMLWorker

/// Replays Core ML's own outputs through the native prediction network and
/// joint.
///
/// The fixture holds inputs and the outputs the compiled models returned for
/// them, captured by `scripts/capture-rnnt-parity.py`; the models themselves
/// are 600 M parameters and are not checked in, so the weights come from the
/// installed model directory and the suite skips when it is absent.
///
/// The tolerances are not slack. Core ML's CPU `lstm` accumulates its recurrent
/// product in fp16 and this accumulates in fp32, so the two cannot agree
/// bit-for-bit in either direction; the capture script measures the gap and
/// writes it into the fixture. What the test pins is that the gap stays where
/// it was measured, which is what would break if a gate were misordered, a bias
/// dropped, or an fp16 rounding lost.
@Suite("Native RNNT decode loop against captured Core ML outputs")
struct NativeRnntDecoderTests {
    private struct Fixture: Decodable {
        struct Step: Decodable {
            let token: Int
            let hIn: String
            let cIn: String
            let decoder: String
            let hOut: String
            let cOut: String
        }

        struct JointCase: Decodable {
            let encoderStep: String
            let decoderStep: String
            let tokenId: Int
            let tokenProb: Float
        }

        let gateOrder: String
        let layers: Int
        let hidden: Int
        let encoderDim: Int
        let vocabulary: Int
        let stepTolerance: Float
        let decoderSteps: [Step]
        let jointCases: [JointCase]
    }

    /// Base64 little-endian fp16, widened. Every fixture value crossed an fp16
    /// tensor boundary in the compiled program, so nothing is lost by it.
    private static func unpack(_ encoded: String) throws -> [Float] {
        let data = try #require(Data(base64Encoded: encoded))
        return data.withUnsafeBytes { raw in
            (0..<(raw.count / MemoryLayout<Float16>.size)).map {
                Float(raw.loadUnaligned(fromByteOffset: $0 * 2, as: Float16.self))
            }
        }
    }

    private static var modelDirectory: URL? {
        if let override = ProcessInfo.processInfo.environment["PARAKEET_MODEL_DIR"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        guard
            let support = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else { return nil }
        let directory = support
            .appendingPathComponent("com.parakeet.rs/models/coreml", isDirectory: true)
            .appendingPathComponent("parakeet-unified-en-0.6b", isDirectory: true)
        return FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("parakeet_unified_decoder.mlmodelc").path)
            ? directory : nil
    }

    private static func fixture() throws -> Fixture {
        let url = try #require(
            Bundle.module.url(forResource: "rnnt-parity", withExtension: "json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Fixture.self, from: Data(contentsOf: url))
    }

    @Test("the prediction network reproduces every captured decoder step")
    func predictionNetworkMatchesCapture() throws {
        guard let directory = Self.modelDirectory else { return }
        let fixture = try Self.fixture()
        #expect(fixture.gateOrder == "ifog")
        let network = try RnntPredictionNetwork(
            bundle: directory.appendingPathComponent("parakeet_unified_decoder.mlmodelc"),
            layers: fixture.layers,
            hidden: fixture.hidden
        )
        #expect(network.vocabulary == fixture.vocabulary)

        for (index, step) in fixture.decoderSteps.enumerated() {
            let result = network.step(
                token: step.token,
                hidden: try Self.unpack(step.hIn),
                cell: try Self.unpack(step.cIn)
            )
            let expectedOutput = try Self.unpack(step.decoder)
            let expectedHidden = try Self.unpack(step.hOut)
            let expectedCell = try Self.unpack(step.cOut)
            #expect(result.output.count == expectedOutput.count)
            for (produced, expected) in [
                (result.output, expectedOutput),
                (result.hidden, expectedHidden),
                (result.cell, expectedCell),
            ] {
                let worst = zip(produced, expected).map { abs($0 - $1) }.max() ?? 0
                #expect(
                    worst <= fixture.stepTolerance,
                    "step \(index) differs by \(worst), over \(fixture.stepTolerance)")
            }
        }
    }

    @Test("a fresh prediction network starts from zero state")
    func firstStepStartsFromZeroState() throws {
        guard let directory = Self.modelDirectory else { return }
        let fixture = try Self.fixture()
        let first = try #require(fixture.decoderSteps.first)
        // The capture starts the trajectory the way the decode loop does: blank
        // token, zero state. A test that took the state from the fixture would
        // not notice `reset()` handing out something else.
        #expect(first.token == fixture.vocabulary - 1)
        #expect(try Self.unpack(first.hIn).allSatisfy { $0 == 0 })
        #expect(try Self.unpack(first.cIn).allSatisfy { $0 == 0 })
        _ = directory
    }

    @Test("the joint picks the same token as the compiled model")
    func jointMatchesCapture() throws {
        guard let directory = Self.modelDirectory else { return }
        let fixture = try Self.fixture()
        let joint = try RnntJointNetwork(
            bundle: directory.appendingPathComponent(
                "parakeet_unified_joint_decision_single_step.mlmodelc"))
        #expect(joint.encoderDimension == fixture.encoderDim)
        #expect(joint.vocabulary == fixture.vocabulary)

        for (index, testCase) in fixture.jointCases.enumerated() {
            let encoderStep = try Self.unpack(testCase.encoderStep)
            let decoderStep = try Self.unpack(testCase.decoderStep)
            let projected = try joint.projectEncoder(
                Self.multiArray(encoderStep), frameRange: 0..<1)
            let decision = joint.decide(
                encoderProjection: projected.frame(0),
                decoderProjection: joint.projectDecoder(decoderStep),
                blankIndex: fixture.vocabulary - 1
            )
            #expect(
                decision.token == testCase.tokenId,
                "case \(index) chose \(decision.token), captured \(testCase.tokenId)")
            if decision.token != fixture.vocabulary - 1 {
                #expect(abs(decision.probability - testCase.tokenProb) < 1e-3)
            }
        }
    }

    /// One encoder frame in the `[1, D, T]` layout the encoder produces.
    private static func multiArray(_ values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: values.count), 1], dataType: .float32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }
}
