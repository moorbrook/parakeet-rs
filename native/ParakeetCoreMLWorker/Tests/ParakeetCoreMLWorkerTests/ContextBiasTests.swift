import Foundation
import Testing

@testable import ParakeetCoreMLWorker

/// The context trie, on synthetic token ids.
///
/// Nothing here loads a model: the arithmetic under test is the graph's, and
/// the ids are arbitrary. `score` is 2 throughout, matching the shipped
/// `hotword_score` default, so the expected numbers read as multiples of it.
@Suite("Contextual biasing context graph")
struct ContextBiasTests {
    private let score: Float = 2

    private func graph(_ entries: [[Int]]) -> ContextGraph {
        ContextGraph(entries: entries, score: score, blankIndex: blank)
    }

    private let blank = 99

    /// The boost the graph would add to one token's logit, or zero.
    private func boost(_ graph: ContextGraph, at state: ContextGraph.Node, token: Int) -> Float {
        graph.bias(from: state).first { $0.token == token }?.boost ?? 0
    }

    private func walk(_ graph: ContextGraph, _ tokens: [Int]) -> ContextGraph.Node {
        var state = graph.root
        for token in tokens {
            state = graph.advance(from: state, token: token)
        }
        return state
    }

    // MARK: Prefix boost

    @Test("the first token of an entry is boosted from the root, and nothing else is")
    func firstTokenIsBoosted() {
        let graph = self.graph([[1, 2, 3]])
        #expect(boost(graph, at: graph.root, token: 1) == score)
        #expect(boost(graph, at: graph.root, token: 2) == 0)
        #expect(boost(graph, at: graph.root, token: 7) == 0)
        // Nothing is at risk at the root, so blank is not touched either.
        #expect(boost(graph, at: graph.root, token: blank) == 0)
        #expect(graph.bias(from: graph.root).count == 1)
    }

    @Test("continuing an entry beats blank by exactly the per-token score")
    func continuingBeatsBlankByTheScore() {
        // The load-bearing relation. Blank does not extend the label sequence,
        // so its true delta is zero and every shifted vector must keep
        // `continue - blank == score`. Getting this wrong — leaving blank at
        // zero alongside the non-matching tokens — makes continuing worth the
        // whole accumulated boost over blank, and the loop spits out the rest
        // of the entry inside one frame regardless of the audio.
        let graph = self.graph([[1, 2, 3]])
        let afterOne = walk(graph, [1])
        #expect(boost(graph, at: afterOne, token: blank) == score)
        #expect(boost(graph, at: afterOne, token: 2) == 2 * score)

        let afterTwo = walk(graph, [1, 2])
        #expect(boost(graph, at: afterTwo, token: blank) == 2 * score)
        #expect(boost(graph, at: afterTwo, token: 3) == 3 * score)
    }

    // MARK: Failure-arc retraction

    @Test("abandoning a partial match retracts exactly what it earned")
    func mismatchRetractsThePartialMatch() {
        let graph = self.graph([[1, 2, 3]])
        let afterTwo = walk(graph, [1, 2])
        // Unshifted: the value a beam would add to the hypothesis.
        #expect(graph.delta(from: afterTwo, token: 7) == -2 * score)
        #expect(graph.delta(from: afterTwo, token: 3) == score)
        #expect(graph.delta(from: afterTwo, token: blank) == 0)
        // Shifted, as the greedy argmax sees it: the mismatch is the zero.
        #expect(boost(graph, at: afterTwo, token: 7) == 0)
    }

    @Test("a mismatch returns the state to the root")
    func mismatchReturnsToRoot() {
        let graph = self.graph([[1, 2, 3]])
        let afterMismatch = walk(graph, [1, 2, 7])
        #expect(afterMismatch === graph.root)
        #expect(graph.bias(from: afterMismatch).count == 1)
    }

    @Test("a mismatch that is itself the start of an entry lands on that entry")
    func mismatchFollowsTheFailureArcIntoAnotherEntry() {
        let graph = self.graph([[1, 2, 3], [7, 8]])
        let afterTwo = walk(graph, [1, 2])
        let landed = graph.advance(from: afterTwo, token: 7)
        #expect(landed.nodeScore == score)
        // Two tokens' worth given back, one earned.
        #expect(graph.delta(from: afterTwo, token: 7) == score - 2 * score)
        #expect(boost(graph, at: afterTwo, token: 7) == score)
        #expect(boost(graph, at: landed, token: 8) == 2 * score)
    }

    @Test("a completed entry is never taken back")
    func completedEntryIsBanked() {
        // The difference from sherpa's beam bookkeeping. After a full match,
        // moving on must cost nothing: an uncompensated retraction sitting on
        // blank would suppress whatever word follows the matched term.
        let graph = self.graph([[1, 2]])
        let matched = walk(graph, [1, 2])
        #expect(matched.isEnd)
        #expect(matched.atRisk == 0)
        #expect(graph.delta(from: matched, token: 7) == 0)
        #expect(boost(graph, at: matched, token: blank) == 0)
        // The root stays on the failure chain, so a *new* entry can start
        // immediately after a matched one — that is the only boost left here.
        #expect(graph.bias(from: matched).map(\.token) == [1])
    }

    // MARK: Entries sharing a prefix

    @Test("a longer entry stays reachable after the shorter one it contains matched")
    func sharedPrefixKeepsExtending() {
        let graph = self.graph([[1, 2], [1, 2, 3]])
        let shorter = walk(graph, [1, 2])
        #expect(shorter.isEnd)
        // Nothing at risk, so blank is untouched, but extending into the longer
        // entry still earns its token.
        #expect(boost(graph, at: shorter, token: blank) == 0)
        #expect(boost(graph, at: shorter, token: 3) == score)

        let longer = walk(graph, [1, 2, 3])
        #expect(longer.isEnd)
        #expect(longer.atRisk == 0)
        #expect(graph.delta(from: longer, token: 7) == 0)
    }

    @Test("a shared prefix is one path, and a shared first token is not counted twice")
    func sharedPrefixIsOnePath() {
        let graph = self.graph([[1, 2], [1, 3]])
        let afterOne = walk(graph, [1])
        #expect(afterOne.nodeScore == score)
        let bias = graph.bias(from: afterOne)
        // 1 is present because restarting the entry is reachable through the
        // failure arc to the root; it is worth exactly what blank is, so
        // repeating the first token neither gains nor loses.
        #expect(Set(bias.map(\.token)) == [blank, 1, 2, 3])
        #expect(boost(graph, at: afterOne, token: 1) == boost(graph, at: afterOne, token: blank))
        #expect(boost(graph, at: afterOne, token: 2) == 2 * score)
        #expect(boost(graph, at: afterOne, token: 3) == 2 * score)
    }

    // MARK: Overlapping entries

    @Test("an entry contained in the middle of another banks its own match")
    func overlappingEntryBanksItsSuffix() {
        // `[1, 2]` is a complete entry and also the tail of `[9, 1, 2, 3]`.
        // Sitting at 9-1-2, the match on `[1, 2]` is done and only the leading
        // 9 is still at risk. Reading the retraction off `nodeScore` alone
        // would give back the completed entry too.
        let graph = self.graph([[9, 1, 2, 3], [1, 2]])
        let inside = walk(graph, [9, 1, 2])
        #expect(inside.nodeScore == 3 * score)
        #expect(inside.endScore == 2 * score)
        #expect(inside.atRisk == score)
        #expect(graph.delta(from: inside, token: 5) == -score)
        #expect(boost(graph, at: inside, token: blank) == score)
        #expect(boost(graph, at: inside, token: 3) == 2 * score)
    }

    @Test("the nearest node wins when a token is reachable at two depths")
    func nearestGotoWins() {
        let graph = self.graph([[1, 2, 3], [2, 4]])
        let afterOne = walk(graph, [1])
        // 2 continues `[1, 2, 3]` from here; the shallower `[2, 4]` start must
        // not override it.
        #expect(graph.advance(from: afterOne, token: 2).nodeScore == 2 * score)
        #expect(boost(graph, at: afterOne, token: 2) == 2 * score)
        #expect(graph.bias(from: afterOne).filter { $0.token == 2 }.count == 1)
    }

    @Test("a duplicated entry does not stack its boost")
    func duplicateEntryDoesNotStack() {
        let graph = self.graph([[1, 2], [1, 2]])
        #expect(boost(graph, at: graph.root, token: 1) == score)
        #expect(walk(graph, [1, 2]).nodeScore == 2 * score)
    }
}

@Suite("Vocabulary term tokenization")
struct PieceVocabularyTests {
    private let pieces = PieceVocabulary(pieces: [
        "\u{2581}N": 1, "e": 2, "w": 3, "\u{2581}Ne": 4, "\u{2581}New": 5,
        "\u{2581}Y": 6, "o": 7, "r": 8, "k": 9, "\u{2581}I": 10, "B": 11, "M": 12,
    ])

    @Test("a term is matched longest-piece-first")
    func longestMatchWins() {
        #expect(pieces.encode("New") == [5])
        #expect(pieces.encode("Newer") == [5, 2, 8])
    }

    @Test("every word is marked word-initial")
    func everyWordIsMarkedWordInitial() {
        // Without the marker on the second word the boost targets the mid-word
        // spelling and the decode renders "NewYork".
        #expect(pieces.encode("New York") == [5, 6, 7, 8, 9])
        #expect(pieces.encode("  New   York  ") == [5, 6, 7, 8, 9])
    }

    @Test("a term with no piece for some part of it is rejected, not truncated")
    func unrepresentableTermIsRejected() {
        // A silently dropped term boosts nothing and the user is never told —
        // the exact failure the sherpa path's token validation exists to catch.
        #expect(pieces.encode("Zzz") == nil)
        #expect(pieces.encode("Newz") == nil)
    }

    @Test("a term with no non-whitespace content encodes to nothing")
    func emptyTermIsRejected() {
        #expect(pieces.encode("   ") == nil)
        #expect(pieces.encode("") == nil)
    }

    @Test("the store reports which terms the inventory could not represent")
    func storeReportsRejections() {
        let store = ContextBiasStore()
        let report = store.set(
            terms: ["IBM", "Zzz", "New York"], score: 2, vocabulary: pieces, blankIndex: 1024)
        #expect(report.accepted == 2)
        #expect(report.rejected == ["Zzz"])
        #expect(store.graph != nil)
    }

    @Test("an empty vocabulary clears biasing rather than building an empty graph")
    func emptyVocabularyClearsTheGraph() {
        let store = ContextBiasStore()
        _ = store.set(terms: ["IBM"], score: 2, vocabulary: pieces, blankIndex: 1024)
        #expect(store.graph != nil)
        let report = store.set(terms: [], score: 2, vocabulary: pieces, blankIndex: 1024)
        #expect(report.accepted == 0)
        #expect(store.graph == nil)
    }
}
