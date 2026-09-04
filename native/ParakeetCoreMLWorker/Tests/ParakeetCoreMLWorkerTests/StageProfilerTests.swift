import CoreML
import Testing

@testable import ParakeetCoreMLWorker

private func event(_ stage: StageProfiler.Stage, _ start: UInt64, _ end: UInt64)
    -> StageProfiler.Event
{
    StageProfiler.Event(stage: stage, startNanoseconds: start, endNanoseconds: end)
}

@Suite("Dispatch overlap accounting")
struct StageProfilerOverlapTests {
    @Test("a serial timeline overlaps by nothing")
    func serialTimelineHasNoOverlap() {
        let timeline = [
            event(.encoder, 0, 100),
            event(.decoder, 100, 150),
            event(.joint, 160, 200),
        ]
        #expect(StageProfiler.overlap(in: timeline) == 0)
    }

    @Test("two dispatches in flight at once are counted twice")
    func concurrentDispatchIsCounted() {
        // 0-100 and 60-160: 40 ns of the 200 ns summed duration is double-counted.
        let timeline = [event(.encoder, 0, 100), event(.encoder, 60, 160)]
        #expect(StageProfiler.overlap(in: timeline) == 40)
    }

    @Test("a dispatch fully inside another counts its whole duration")
    func nestedDispatchCountsEntirely() {
        let timeline = [event(.encoder, 0, 100), event(.joint, 20, 50)]
        #expect(StageProfiler.overlap(in: timeline) == 30)
    }

    @Test("overlap accumulates across separate concurrent groups")
    func separateGroupsBothCount() {
        let timeline = [
            event(.encoder, 0, 100),
            event(.joint, 50, 120),   // 50 ns of overlap
            event(.decoder, 500, 600),
            event(.joint, 550, 700),  // 50 ns of overlap
        ]
        #expect(StageProfiler.overlap(in: timeline) == 100)
    }

    /// `overlap(in:)` merges forward in one pass, so it is only correct on a
    /// timeline sorted by start. `endUtterance` sorts before reducing; this
    /// pins that the sort is load-bearing rather than incidental tidiness.
    @Test("the reduction depends on the timeline being sorted by start")
    func sortOrderIsLoadBearing() {
        let sorted = [event(.encoder, 0, 100), event(.joint, 60, 160)]
        let unsorted = [event(.joint, 60, 160), event(.encoder, 0, 100)]
        #expect(StageProfiler.overlap(in: sorted) == 40)
        #expect(StageProfiler.overlap(in: unsorted) != 40)
    }

    @Test("an empty timeline overlaps by nothing")
    func emptyTimeline() {
        #expect(StageProfiler.overlap(in: []) == 0)
    }
}

@Suite("Stage timeline reduction")
struct StageProfilerReduceTests {
    /// Parakeet Unified: mel is Swift, so there is no preprocessor dispatch and
    /// the gap before the encoder is the mel cost.
    @Test("a Unified window attributes the pre-encoder gap to mel")
    func unifiedShape() {
        let report = StageProfiler.reduce(
            timeline: [
                event(.encoder, 3_000_000, 28_000_000),
                event(.decoder, 28_000_000, 30_000_000),
                event(.joint, 30_000_000, 33_000_000),
            ],
            startNanoseconds: 0,
            endNanoseconds: 34_000_000,
            computeUnits: "test"
        )
        #expect(report.preprocessorCalls == 0)
        #expect(report.preprocessorMs == 0)
        #expect(report.encoderCalls == 1)
        #expect(report.windows == 1)
        #expect(report.melMs == 3)
        #expect(report.encoderMs == 25)
        #expect(report.decodeLoopMs == 5)
        #expect(report.postMs == 1)
        #expect(report.overlappedDispatchMs == 0)
        // The stage split partitions the profiled interval.
        #expect(
            report.melMs + report.preprocessorMs + report.encoderMs + report.decodeLoopMs
                + report.postMs == report.totalMs)
    }

    /// TDT: mel is its own Core ML graph, so it must land in `preprocessorMs`
    /// rather than vanishing into the gap the way it did before that stage
    /// existed.
    @Test("a TDT window bills the mel graph to the preprocessor stage")
    func tdtShape() {
        let report = StageProfiler.reduce(
            timeline: [
                event(.preprocessor, 1_000_000, 3_000_000),
                event(.encoder, 3_000_000, 28_000_000),
                event(.decoder, 28_000_000, 30_000_000),
                event(.joint, 30_000_000, 33_000_000),
            ],
            startNanoseconds: 0,
            endNanoseconds: 34_000_000,
            computeUnits: "test"
        )
        #expect(report.preprocessorCalls == 1)
        #expect(report.preprocessorMs == 2)
        #expect(report.melMs == 1)
        #expect(report.encoderMs == 25)
        #expect(report.decodeLoopMs == 5)
        #expect(report.postMs == 1)
        #expect(
            report.melMs + report.preprocessorMs + report.encoderMs + report.decodeLoopMs
                + report.postMs == report.totalMs)
    }

    /// A second window's mel dispatch closes the first window's decode loop
    /// instead of extending it.
    @Test("a mel dispatch closes the previous window's decode loop")
    func twoWindowsSplitTheDecodeLoop() {
        let report = StageProfiler.reduce(
            timeline: [
                event(.preprocessor, 0, 1_000_000),
                event(.encoder, 1_000_000, 11_000_000),
                event(.joint, 11_000_000, 13_000_000),
                event(.preprocessor, 14_000_000, 15_000_000),
                event(.encoder, 15_000_000, 25_000_000),
                event(.joint, 25_000_000, 26_000_000),
            ],
            startNanoseconds: 0,
            endNanoseconds: 27_000_000,
            computeUnits: "test"
        )
        #expect(report.windows == 2)
        #expect(report.preprocessorCalls == 2)
        #expect(report.encoderMs == 20)
        // 2 ms after the first window plus 1 ms after the second. The 1 ms gap
        // before the second mel dispatch is mel, not decode loop.
        #expect(report.decodeLoopMs == 3)
        #expect(report.melMs == 1)
        #expect(
            report.melMs + report.preprocessorMs + report.encoderMs + report.decodeLoopMs
                + report.postMs == report.totalMs)
    }

    @Test("concurrent chunk decode is reported rather than silently double-counted")
    func concurrentDecodeIsFlagged() {
        let report = StageProfiler.reduce(
            timeline: [
                event(.encoder, 0, 10_000_000),
                event(.joint, 10_000_000, 20_000_000),
                event(.joint, 15_000_000, 25_000_000),
            ],
            startNanoseconds: 0,
            endNanoseconds: 25_000_000,
            computeUnits: "test"
        )
        #expect(report.overlappedDispatchMs == 5)
    }
}

@Suite("Dispatch stage classification")
struct StageProfilerClassifyTests {
    private func provider(_ names: [String]) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [:]
        for name in names {
            features[name] = MLFeatureValue(int64: 0)
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    @Test("the encoder is identified by its mel input in both graph sets")
    func encoderInputs() throws {
        #expect(StageProfiler.classify(try provider(["mel", "mel_length"])) == .encoder)
    }

    @Test("TDT's mel graph is identified by its waveform input")
    func preprocessorInputs() throws {
        #expect(
            StageProfiler.classify(try provider(["audio_signal", "audio_length"]))
                == .preprocessor)
    }

    @Test("the joint is identified by its encoder and decoder steps")
    func jointInputs() throws {
        #expect(
            StageProfiler.classify(try provider(["encoder_step", "decoder_step"])) == .joint)
    }

    @Test("the decoder is identified by its target tokens")
    func decoderInputs() throws {
        #expect(
            StageProfiler.classify(try provider(["targets", "target_length", "h_in", "c_in"]))
                == .decoder)
    }

    /// The encoder consumes the mel graph's output, so both carry mel-ish
    /// names; `mel` must win over `audio_signal` or every TDT encoder dispatch
    /// would be billed to the preprocessor.
    @Test("mel wins over the waveform when a provider carries both")
    func melTakesPrecedence() throws {
        #expect(
            StageProfiler.classify(try provider(["mel", "audio_signal"])) == .encoder)
    }

    @Test("an unrecognised provider is not silently attributed to a stage")
    func unknownInputs() throws {
        #expect(StageProfiler.classify(try provider(["something_else"])) == .other)
    }
}
