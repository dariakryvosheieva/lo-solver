import heapq
from features import tokenize


class Segmenter:
    """
    Helper class to generate plausible segmentations of the Problemese input for a given
    part of speech.
    """

    def __init__(
        self,
        allow_empty_slots,
        segmentations_beyond_minimal,
        segmentation_queue_limit,
        segmentation_queue_trim_factor,
        words,
        word_to_present_slots,
        word_to_slot_values,
        ed,
        debug,
    ):
        self.allow_empty_slots = allow_empty_slots  # whether to allow empty realizations of present morphemes
        self.segmentations_beyond_minimal = segmentations_beyond_minimal  # keep all segmentations that minimize TFED plus K next-best ones
        self.segmentation_queue_limit = segmentation_queue_limit  # trim suboptimal segmentations to prevent OOM
        self.segmentation_queue_trim_factor = segmentation_queue_trim_factor  # once queue size reaches limit * trim_factor, trim to limit
        self.words = words
        self.word_to_present_slots = word_to_present_slots
        self.word_to_slot_values = word_to_slot_values
        self.ed = ed
        self.debug = debug

        # Tokenization cache: avoid re-tokenizing to save time
        self._token_cache = {}
        for w, _feats in self.words:
            try:
                self._token_cache[w] = tuple(tokenize(w))
            except Exception:
                # Fall back to per-call tokenization if something unexpected happens
                self._token_cache[w] = None

    def _tokens(self, surface: str):
        """Return cached phoneme tokens for a surface string (or tokenize on cache miss)."""
        toks = self._token_cache.get(surface)
        if toks is None:
            toks = tuple(tokenize(surface))
            self._token_cache[surface] = toks
        return toks

    def get_segments(self, word, order, breaks):
        """Extract segments from a word given break positions."""
        surface = word[0]
        toks = self._tokens(surface)
        segments = {}
        # breaks are phoneme-token indices, not character offsets
        breaks_with_end = breaks + [len(toks)]
        for i, slot in enumerate(order):
            start = breaks_with_end[i]
            end = breaks_with_end[i + 1]
            segments[slot] = "".join(toks[start:end])
        return segments

    def _compositions(self, n, k, positive):
        """Generate all possible tuples of k non-negative (or positive) integers summing to n."""
        if k == 0:
            if n == 0:
                yield tuple()
            return
        if positive and n < k:
            return

        def _rec(rem, parts, prefix):
            if parts == 1:
                yield tuple(prefix + [rem])
                return
            start, end = (1, rem - parts + 2) if positive else (0, rem + 1)
            for i in range(start, end):
                yield from _rec(rem - i, parts - 1, prefix + [i])

        yield from _rec(n, k, [])

    def _generate_break_candidates(self, word, order):
        """Generate candidate morpheme break positions."""
        present = self.word_to_present_slots.get(word, {"root"})
        indices_present = [idx for idx, s in enumerate(order) if s in present]
        K = len(order)
        P = len(indices_present)
        L = len(self._tokens(word[0]))
        positive = not self.allow_empty_slots

        candidates = []
        for compP in self._compositions(L, P, positive=positive):
            compK = [0] * K
            for idx, ln in zip(indices_present, compP):
                compK[idx] = ln
            breaks = [0]
            for ln in compK:
                breaks.append(breaks[-1] + ln)
            candidates.append(breaks[:-1])

        return candidates

    def _copy_seg_stats(self, stats):
        """Deep-copy seg_stats."""
        return {
            k: {
                "segments": v["segments"][:],
                "pairwise_sum": v.get("pairwise_sum", 0.0),
                "mst_weight": v.get("mst_weight", 0.0),
                "mst_edges": v.get("mst_edges", [])[:],
            }
            for k, v in stats.items()
        }

    def _mst_recompute(self, num_nodes, edges):
        """Compute MST weight and chosen edges via Kruskal's algorithm."""
        if num_nodes <= 1:
            return 0.0, []

        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        mst_edges = []
        mst_weight = 0.0
        for w, u, v in sorted(edges, key=lambda t: t[0]):
            if union(u, v):
                mst_edges.append((w, u, v))
                mst_weight += float(w)
                if len(mst_edges) == num_nodes - 1:
                    break

        return mst_weight, mst_edges

    def _trim_segmentation_queue(self, pq, limit):
        """Drop the worst-scoring segmentation states when the queue grows too large."""
        best_states = heapq.nsmallest(limit, pq)
        heapq.heapify(best_states)
        return best_states

    def generate_segmentations(
        self,
        order,
    ):
        """
        Generate segmentations ranked by total feature edit distance (TFED) using BFS.

        We define TFED as the sum, over all (slot, value) keys i, of TFED_i, where TFED_i
        is the MST feature edit distance of realizations of key i in the current segmentation.
        """
        all_segmentations = []
        best_seg_score = float("inf")

        # Priority queue over states: (current_tfed, counter, word_idx, segmentations, seg_stats)
        pq = []
        counter = 0

        heapq.heappush(pq, (0.0, counter, 0, {}, {}))
        counter += 1

        best_tfed = None
        kept_segmentations = []
        kept_complete_keys = set()
        kept_next = 0

        while pq:
            tfed, _, idx, seg_dict, stats = heapq.heappop(pq)

            # All words segmented: record this complete segmentation
            if idx == len(self.words):
                if best_tfed is None:
                    best_tfed = tfed

                # Canonicalize to avoid accidental duplicates in the output list
                complete_key = tuple(
                    sorted((w, tuple(brks)) for (w, brks) in seg_dict.items())
                )
                if complete_key in kept_complete_keys:
                    continue

                if tfed == best_tfed:
                    kept_complete_keys.add(complete_key)
                    kept_segmentations.append((dict(seg_dict), tfed))
                    continue

                # Keep up to K next-best complete segmentations beyond the TFED-minimal set
                if kept_next < self.segmentations_beyond_minimal:
                    kept_complete_keys.add(complete_key)
                    kept_segmentations.append((dict(seg_dict), tfed))
                    kept_next += 1
                if kept_next >= self.segmentations_beyond_minimal:
                    break
                continue

            # Segment the next word
            w = self.words[idx]
            vals = self.word_to_slot_values.get(w, {})

            break_candidates = self._generate_break_candidates(w, order)

            for breaks in break_candidates:
                seg_word = self.get_segments(w, order, breaks)

                # Compute incremental contribution to TFED and updated seg_stats
                delta_tfed = 0.0
                new_stats = self._copy_seg_stats(stats)
                for slot in order:
                    if slot not in vals:
                        continue
                    key = (slot, vals[slot])
                    seg_piece = seg_word.get(slot, "")

                    sstats = new_stats.setdefault(
                        key, {"segments": [], "mst_weight": 0.0, "mst_edges": []}
                    )
                    old_segments = sstats["segments"]
                    old_w = float(sstats.get("mst_weight", 0.0))
                    old_edges = sstats.get("mst_edges", [])

                    # Append the new node; compute all edges from new node to existing nodes
                    new_idx = len(old_segments)
                    new_edges = []
                    for i, existing_seg in enumerate(old_segments):
                        dist = float(self.ed.edit_distance(seg_piece, existing_seg))
                        new_edges.append((dist, i, new_idx))

                    # Candidate edge set: previous MST edges plus new-node edges
                    cand_edges = list(old_edges) + new_edges
                    new_w, new_mst_edges = self._mst_recompute(new_idx + 1, cand_edges)

                    sstats["segments"] = old_segments + [seg_piece]
                    sstats["mst_weight"] = new_w
                    sstats["mst_edges"] = new_mst_edges
                    delta_tfed += new_w - old_w

                new_tfed = tfed + delta_tfed
                new_segmentation = dict(seg_dict)
                new_segmentation[w] = breaks
                heapq.heappush(
                    pq,
                    (
                        new_tfed,
                        counter,
                        idx + 1,
                        new_segmentation,
                        new_stats,
                    ),
                )
                counter += 1

                if (
                    len(pq)
                    > self.segmentation_queue_limit
                    * self.segmentation_queue_trim_factor
                ):
                    pq = self._trim_segmentation_queue(
                        pq, self.segmentation_queue_limit
                    )

        if best_tfed is not None:
            best_seg_score = best_tfed
        all_segmentations.extend(kept_segmentations)

        if self.debug:
            min_count = 0
            if best_tfed is not None:
                min_count = sum(1 for (_, s) in kept_segmentations if s == best_tfed)
            print(
                f"Generated {len(all_segmentations)} segmentations "
                f"({min_count} minimal, +{max(0, len(all_segmentations) - min_count)} next-best; "
                f"TFED_min={best_seg_score})",
            )

        return all_segmentations
