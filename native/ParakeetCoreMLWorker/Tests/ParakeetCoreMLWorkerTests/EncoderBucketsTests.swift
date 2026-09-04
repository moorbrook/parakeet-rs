import Testing

@testable import ParakeetCoreMLWorker

@Suite("Encoder bucket routing")
struct EncoderBucketsTests {
    private let windows = [2, 5, 8]

    @Test("an utterance goes to the narrowest window that holds it")
    func picksTheNarrowestFittingWindow() {
        #expect(EncoderBuckets.select(from: windows, sampleCount: 1) == 2)
        #expect(EncoderBuckets.select(from: windows, sampleCount: 16_000) == 2)
        #expect(EncoderBuckets.select(from: windows, sampleCount: 48_000) == 5)
        #expect(EncoderBuckets.select(from: windows, sampleCount: 100_000) == 8)
    }

    @Test("a window holds an utterance exactly its own length")
    func boundaryIsInclusive() {
        #expect(EncoderBuckets.select(from: windows, sampleCount: 32_000) == 2)
        #expect(EncoderBuckets.select(from: windows, sampleCount: 32_001) == 5)
        #expect(EncoderBuckets.select(from: windows, sampleCount: 128_000) == 8)
    }

    @Test("an utterance longer than every window falls through to the caller")
    func tooLongFallsThrough() {
        #expect(EncoderBuckets.select(from: windows, sampleCount: 128_001) == nil)
    }

    @Test("routing does not depend on the order windows were given in")
    func unsortedWindowsRouteTheSame() {
        #expect(EncoderBuckets.select(from: [8, 2, 5], sampleCount: 48_000) == 5)
    }

    @Test("with no buckets every utterance falls through")
    func noBucketsFallThrough() {
        #expect(EncoderBuckets.select(from: [], sampleCount: 1) == nil)
    }
}
