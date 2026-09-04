import Foundation

/// Shallow-fusion contextual biasing over the native joint's output.
///
/// The user's custom vocabulary used to force the whole app onto the
/// sherpa-onnx fallback, because that backend owned the only contextual-biasing
/// implementation (ADR-0020, ADR-0022). Sherpa's `modified_beam_search`
/// hotword path builds an Aho-Corasick context graph over the hotwords'
/// token ids and adds the graph's score delta to each hypothesis before the
/// beam is pruned. This is the same graph, applied to the greedy loop's joint
/// logits before the argmax, so the biased path keeps the int8 encoder on the
/// Neural Engine.
///
/// ## What the numbers mean
///
/// `nodeScore` is the boost accumulated along the trie path to a node: `depth *
/// score` for a uniform per-token score. `endScore` is the part of that which a
/// completed entry has already earned and which must never be taken back —
/// the deepest entry ending on the path to the node, or, for a node whose
/// *suffix* is a complete entry, the score of that suffix (the Aho-Corasick
/// output link). The difference, `atRisk = nodeScore - endScore`, is what a
/// hypothesis loses by walking away mid-entry.
///
/// A step's true shallow-fusion deltas from state `s`, for the joint's three
/// kinds of outcome, are then:
///
/// - blank: `0`. Blank does not extend the label sequence, so the context state
///   and its accumulated boost are unchanged.
/// - a token continuing (or starting) an entry, reaching node `n`:
///   `atRisk(n) + banked(s, n) - atRisk(s)`.
/// - any other token: `-atRisk(s)`, the retraction — the failure arc lands on
///   the root (or on the node for the longest matching suffix) and the
///   unfinished entry's boost is given back.
///
/// A greedy argmax only compares outcomes within one step, so the whole vector
/// may be shifted by any constant. Shifting by `+atRisk(s)` puts the
/// non-matching tokens at zero, which is what makes the boost sparse: only
/// blank and the tokens reachable from `s` or its failure chain need touching.
/// [`ContextGraph.bias(from:)`] returns exactly those, and
/// [`ContextGraph.delta(from:token:)`] returns the unshifted value, which is
/// negative on a mismatch and is what the retraction tests assert on.
///
/// ## Where it differs from sherpa
///
/// Sherpa keeps a completed entry's `nodeScore` at risk until `Finalize` and
/// compensates with a separate `output_score` bonus. That is invisible in a
/// beam, where the two cancel along every complete path, but a greedy decoder
/// takes one step at a time: an uncompensated `+nodeScore` sitting on blank
/// after a completed entry would suppress the word that follows it. Folding the
/// banked part into `endScore` gives a state at a completed entry an `atRisk`
/// of zero, so finishing an entry costs the next word nothing.
final class ContextGraph {
    /// One trie node. Children are owned; `fail` is a back edge into the same
    /// trie and must not retain.
    final class Node {
        let token: Int
        /// Boost on the edge into this node.
        fileprivate(set) var tokenScore: Float = 0
        /// Sum of `tokenScore` from the root to here.
        fileprivate(set) var nodeScore: Float = 0
        /// The part of `nodeScore` a completed entry has already earned.
        fileprivate(set) var endScore: Float = 0
        /// An entry ends here.
        fileprivate(set) var isEnd = false
        fileprivate(set) var children: [Int: Node] = [:]
        fileprivate unowned var fail: Node!

        fileprivate init(token: Int) {
            self.token = token
        }

        /// The boost still exposed to retraction if the decode leaves this node.
        var atRisk: Float { nodeScore - endScore }
    }

    /// A single token's boost, applied to one joint logit before the argmax.
    struct BiasEntry {
        let token: Int
        let boost: Float
    }

    let root = Node(token: -1)
    let blankIndex: Int
    /// Entries whose tokenization succeeded, for the readiness report.
    private(set) var acceptedCount = 0

    /// Build the graph over already-tokenized entries.
    ///
    /// `score` is the per-token boost, the same knob as sherpa's
    /// `hotwords_score`. It is uniform across entries: a per-entry score would
    /// need every descendant's `nodeScore` recomputed when a shared prefix's
    /// edge score rises, and nothing in the product sets one.
    init(entries: [[Int]], score: Float, blankIndex: Int) {
        self.blankIndex = blankIndex
        root.fail = root
        for entry in entries where !entry.isEmpty {
            acceptedCount += 1
            var node = root
            for (index, token) in entry.enumerated() {
                let child: Node
                if let existing = node.children[token] {
                    child = existing
                } else {
                    child = Node(token: token)
                    child.tokenScore = score
                    child.nodeScore = node.nodeScore + score
                    node.children[token] = child
                }
                if index == entry.count - 1 {
                    child.isEnd = true
                }
                node = child
            }
        }
        fillFailureArcsAndBanks()
    }

    /// Breadth-first, so a node's failure target and its trie parent are both
    /// finished before the node itself is.
    private func fillFailureArcsAndBanks() {
        var queue: [Node] = []
        for child in root.children.values {
            child.fail = root
            child.endScore = child.isEnd ? child.nodeScore : 0
            queue.append(child)
        }
        var head = 0
        while head < queue.count {
            let node = queue[head]
            head += 1
            for (token, child) in node.children {
                child.fail = failureTarget(from: node.fail, token: token)
                // A completed entry banks its own score. Otherwise the node
                // inherits whatever its trie parent had banked, or — when its
                // own suffix is a completed entry, which is what the failure
                // chain finds — that suffix's banked score.
                var banked = child.isEnd ? child.nodeScore : node.endScore
                banked = max(banked, child.fail.endScore)
                // A suffix cannot be worth more than the whole path to here.
                child.endScore = min(banked, child.nodeScore)
                queue.append(child)
            }
        }
    }

    /// Aho-Corasick goto over the failure chain: the deepest node whose path is
    /// a suffix of `start`'s path followed by `token`, or the root.
    private func failureTarget(from start: Node, token: Int) -> Node {
        var node = start
        while true {
            if let child = node.children[token] {
                return child
            }
            if node === root {
                return root
            }
            node = node.fail
        }
    }

    /// The state after emitting `token` from `state`.
    ///
    /// Blank never reaches here: it does not extend the label sequence, so it
    /// leaves the context state alone.
    func advance(from state: Node, token: Int) -> Node {
        if let child = state.children[token] {
            return child
        }
        return failureTarget(from: state.fail, token: token)
    }

    /// The unshifted shallow-fusion score change of emitting `token`.
    ///
    /// Positive while an entry is being matched, negative when a partial match
    /// is abandoned, and zero for a token that was not part of any match. This
    /// is the value a beam would add to a hypothesis; the greedy loop uses
    /// [`bias(from:)`], which is this vector shifted so the common case is zero.
    func delta(from state: Node, token: Int) -> Float {
        if token == blankIndex {
            return 0
        }
        let next = advance(from: state, token: token)
        return boost(from: state, to: next) - state.atRisk
    }

    /// The sparse per-token boost to add to the joint logits at `state`.
    ///
    /// Every token absent from the result gets zero. The result is small: the
    /// blank plus one entry per token reachable from `state` or its failure
    /// chain.
    func bias(from state: Node) -> [BiasEntry] {
        var entries: [BiasEntry] = []
        if state.atRisk > 0 {
            entries.append(BiasEntry(token: blankIndex, boost: state.atRisk))
        }
        var seen: Set<Int> = [blankIndex]
        var node: Node? = state
        while let current = node {
            for (token, child) in current.children where !seen.contains(token) {
                seen.insert(token)
                let boost = boost(from: state, to: child)
                if boost != 0 {
                    entries.append(BiasEntry(token: token, boost: boost))
                }
            }
            node = current === root ? nil : current.fail
        }
        return entries
    }

    /// The shifted boost for a step that lands on `next`: what is still at risk
    /// there, plus whatever the step newly banked.
    private func boost(from state: Node, to next: Node) -> Float {
        next.atRisk + max(0, next.endScore - state.endScore)
    }
}

/// The model's SentencePiece pieces, read from the bundle's `vocab.json`.
///
/// Terms are boosted as token sequences, so the app's plain-text vocabulary has
/// to be tokenized with the same inventory the joint emits from. That
/// inventory ships beside the compiled models as an id → piece map.
struct PieceVocabulary {
    private let ids: [String: Int]
    private let longestPiece: Int

    enum LoadError: LocalizedError {
        case unreadable(URL, String)
        case empty(URL)

        var errorDescription: String? {
            switch self {
            case .unreadable(let url, let reason):
                "cannot read the token inventory at \(url.path): \(reason)"
            case .empty(let url):
                "the token inventory at \(url.path) has no pieces"
            }
        }
    }

    /// SentencePiece's word-start marker. A term boosted without it biases the
    /// mid-word spelling and renders glued to the preceding word.
    static let wordStart = "\u{2581}"

    init(vocabularyFile: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: vocabularyFile)
        } catch {
            throw LoadError.unreadable(vocabularyFile, error.localizedDescription)
        }
        guard
            let raw = try? JSONSerialization.jsonObject(with: data) as? [String: String]
        else {
            throw LoadError.unreadable(vocabularyFile, "not a JSON object of id → piece")
        }
        var ids: [String: Int] = [:]
        var longest = 0
        for (id, piece) in raw {
            guard let id = Int(id), !piece.isEmpty else { continue }
            // Duplicate pieces would be a broken inventory; the lowest id wins
            // so the mapping is at least deterministic.
            if let existing = ids[piece], existing <= id { continue }
            ids[piece] = id
            longest = max(longest, piece.count)
        }
        guard !ids.isEmpty else { throw LoadError.empty(vocabularyFile) }
        self.ids = ids
        longestPiece = longest
    }

    init(pieces: [String: Int]) {
        ids = pieces
        longestPiece = pieces.keys.map(\.count).max() ?? 0
    }

    /// Tokenize one vocabulary term, or return `nil` when some part of it has
    /// no piece at all.
    ///
    /// Longest-match from the left. The bundle ships the piece inventory but
    /// not the merge ranks a faithful BPE encoder needs, so this is an
    /// approximation of the segmentation the model itself would produce; it
    /// agrees with it on the ordinary case of a word built from whole pieces,
    /// and a disagreement costs biasing on that term rather than correctness.
    ///
    /// A `nil` is the case the user has to be told about: the term is boosting
    /// nothing, and on the sherpa path the equivalent silent drop is exactly
    /// what `crate::vocabulary`'s token validation exists to catch.
    func encode(_ term: String) -> [Int]? {
        var characters: [Character] = []
        for word in term.split(whereSeparator: \.isWhitespace) {
            characters.append(contentsOf: Self.wordStart)
            characters.append(contentsOf: word)
        }
        guard !characters.isEmpty else { return nil }

        var tokens: [Int] = []
        var index = 0
        while index < characters.count {
            var matched = false
            let limit = min(longestPiece, characters.count - index)
            var length = limit
            while length >= 1 {
                let piece = String(characters[index..<(index + length)])
                if let id = ids[piece] {
                    tokens.append(id)
                    index += length
                    matched = true
                    break
                }
                length -= 1
            }
            if !matched { return nil }
        }
        return tokens
    }
}

/// The vocabulary a running worker is biasing toward.
///
/// Held by reference and read by every decoder instance, because the encoder
/// buckets mean several `UnifiedAsrManager`s over one model directory and the
/// vocabulary arrives after all of them have loaded. Mutated only by the
/// worker's request loop, which is single-threaded, but decoders may read it
/// from a Core ML continuation, so the access is locked.
final class ContextBiasStore: @unchecked Sendable {
    struct Report {
        let accepted: Int
        let rejected: [String]
    }

    private let lock = NSLock()
    private var storedGraph: ContextGraph?
    private var storedBlankIndex: Int?

    /// The blank id of the loaded model, recorded by the decoder factory rather
    /// than assumed, so a model whose blank moved cannot be biased against the
    /// wrong logit. Nil until a native decoder has been built.
    var blankIndex: Int? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return storedBlankIndex
        }
        set {
            lock.lock()
            storedBlankIndex = newValue
            lock.unlock()
        }
    }

    /// The graph in force, or nil when the vocabulary is empty — which is the
    /// default, and costs the decode loop one nil check per utterance.
    var graph: ContextGraph? {
        lock.lock()
        defer { lock.unlock() }
        return storedGraph
    }

    /// Replace the vocabulary. An empty `terms` clears biasing.
    func set(
        terms: [String], score: Float, vocabulary: PieceVocabulary, blankIndex: Int
    ) -> Report {
        var entries: [[Int]] = []
        var rejected: [String] = []
        for term in terms {
            if let tokens = vocabulary.encode(term), !tokens.isEmpty {
                entries.append(tokens)
            } else {
                rejected.append(term)
            }
        }
        let graph =
            entries.isEmpty
            ? nil : ContextGraph(entries: entries, score: score, blankIndex: blankIndex)
        lock.lock()
        storedGraph = graph
        lock.unlock()
        return Report(accepted: entries.count, rejected: rejected)
    }
}
