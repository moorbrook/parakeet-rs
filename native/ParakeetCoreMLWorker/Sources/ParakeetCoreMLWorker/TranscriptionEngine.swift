import CoreML
import FluidAudio
import Foundation

/// Which Parakeet graph set the worker loads.
///
/// `unified` is the shipping default (ADR-0022). `tdtV3` exists to measure the
/// multilingual TDT 0.6B v3 conversion against it (kata f0zg) and is not a
/// production path: it has no Rust-managed integrity gate, so it refuses
/// `--model-root` and forbids FluidAudio's downloader outright.
enum ModelVariant: String {
    case unified = "unified"
    case tdtV3 = "tdt-v3"

    static func parse(_ value: String) throws -> Self {
        guard let variant = Self(rawValue: value) else {
            throw WorkerError.invalidArgument(
                "--model-variant must be unified or tdt-v3, not \(value)"
            )
        }
        return variant
    }

    /// FluidAudio derives this from `Repo.folderName`, and `AsrModels.load`
    /// re-appends it to the parent of the directory it is handed, so the
    /// model directory has to be named exactly this.
    var folderName: String {
        switch self {
        case .unified: "parakeet-unified-en-0.6b"
        case .tdtV3: "parakeet-tdt-0.6b-v3"
        }
    }
}

/// One loaded graph set, behind the single call the worker loop makes.
///
/// An enum rather than a protocol conformance so neither FluidAudio type has
/// to be extended: `UnifiedAsrManager` and `AsrManager` are both actors in a
/// pinned external package, and `EncoderBuckets` keeps handing back the
/// concrete unified manager.
enum TranscriptionEngine {
    case unified(UnifiedAsrManager)
    case tdt(TdtSession)

    func transcribe(_ samples: [Float]) async throws -> String {
        switch self {
        case .unified(let manager): try await manager.transcribe(samples)
        case .tdt(let session): try await session.transcribe(samples)
        }
    }
}

/// A TDT manager plus the per-utterance decoder state it requires.
///
/// `AsrManager.transcribe` takes the LSTM prediction-network state `inout` so
/// a caller can carry it across a stream. Dictation utterances are
/// independent, so every call starts from a zeroed state; keeping a state
/// across utterances would leak the previous transcript's context into the
/// next one and make repeated bench runs non-reproducible.
actor TdtSession {
    private let manager: AsrManager
    private let decoderLayers: Int

    init(manager: AsrManager, decoderLayers: Int) {
        self.manager = manager
        self.decoderLayers = decoderLayers
    }

    func transcribe(_ samples: [Float]) async throws -> String {
        var state = try TdtDecoderState(decoderLayers: decoderLayers)
        let result = try await manager.transcribe(samples, decoderState: &state)
        return result.text
    }
}
