import itertools, heapq, traceback
from collections import defaultdict, Counter

from morph import Morph
from features import FeatureBank, tokenize

from order_scorer import OrderScorer
from edit_distance_calculator import EditDistanceCalculator
from segmenter import Segmenter
from rule_search import RuleSolver, _align_ops, can_invert_rule, invert_rule
from rule_application import apply_rule
from rule import BoundarySpecification


class MorphemeSolver:
    """Solver for morphemes and phonological rules for a given part-of-speech."""

    def __init__(
        self,
        matrix_rows,
        word_features,
        max_total_cost=35.0,
        max_rule_cost=10.0,
        k=1,
        allow_empty_slots=True,
        segmentations_beyond_minimal=0,
        segmentation_queue_limit=100000,
        segmentation_queue_trim_factor=1.5,
        initial_lexicon_max_states=float("inf"),
        debug=True,
    ):
        self.matrix_rows = matrix_rows
        self.word_features = word_features
        self.max_total_cost = max_total_cost  # max total cost of rules in a solution
        self.max_rule_cost = max_rule_cost  # max cost of a single rule
        self.k = k  # number of candidate rules returned by rule search
        self.initial_lexicon_max_states = initial_lexicon_max_states  # number of initial morpheme UR lexica to consider
        self.DEBUG = debug  # enable debug output

        # Extract words and metadata
        self.slots = sorted({slot for f in word_features for (slot, _) in f})
        self.base_list = sorted({f[0][1] for f in word_features if f[0][0] == "root"})
        self.word_features_no_root = sorted(set(f[1:] for f in word_features))

        self.words = []
        for i, b in enumerate(self.base_list):
            row = matrix_rows[i]
            for j, w in enumerate(row):
                if w is None:
                    continue
                feats_no_root = self.word_features_no_root[j]
                full_feats = (("root", b),) + tuple(feats_no_root)
                self.words.append((w, full_feats))

        self.surface_words = [w for (w, _) in self.words]
        self.bank = FeatureBank(self.surface_words)

        self.word_to_present_slots = {}
        self.word_to_slot_values = {}
        self.slot_to_value_to_words = defaultdict(lambda: defaultdict(list))

        for i, b in enumerate(self.base_list):
            row = matrix_rows[i]
            for j, w in enumerate(row):
                if w is None:
                    continue
                feats_no_root = self.word_features_no_root[j]
                full_feats = (("root", b),) + tuple(feats_no_root)
                wkey = (w, full_feats)
                present = {"root"} | {slot for (slot, _) in feats_no_root}
                self.word_to_present_slots[wkey] = set(present)
                vals = {"root": b}
                for slot, v in feats_no_root:
                    vals[slot] = v
                self.word_to_slot_values[wkey] = vals
                for slot, v in vals.items():
                    self.slot_to_value_to_words[slot][v].append(wkey)

        # Helper classes
        self.order_scorer = OrderScorer(
            self.surface_words,
            self.word_to_present_slots,
            self.slot_to_value_to_words,
        )
        self.ed = EditDistanceCalculator(self.bank)
        self.segmenter = Segmenter(
            allow_empty_slots,
            segmentations_beyond_minimal,
            segmentation_queue_limit,
            segmentation_queue_trim_factor,
            self.words,
            self.word_to_present_slots,
            self.word_to_slot_values,
            self.ed,
            self.DEBUG,
        )

    def _predict_sr_for_word(
        self, word, vals, order, morphemes, global_rules, local_rules
    ):
        """Predict the the word's surface form under the current theory."""
        # Build expected UR
        expected_ur_parts = []
        for slot in order:
            if slot not in vals:
                continue
            key = (slot, vals[slot])
            part = morphemes[key]
            expected_ur_parts.append(part)
        expected_ur = "".join(expected_ur_parts)

        # Build expected SR by applying existing rules
        if global_rules or local_rules:
            try:
                # Track morpheme boundary changes across sequential rule applications
                present_slots_in_order = [s for s in order if s in vals]
                keys_in_word = [(s, vals[s]) for s in present_slots_in_order]
                ends = []
                pos = 0
                for key in keys_in_word:
                    part = morphemes.get(key, "")
                    part_len = len(Morph(part)) if part else 0
                    pos += part_len
                    ends.append(pos)

                def _update_ends(ends_list, trace):
                    op = trace.get("op")
                    if op == "delete":
                        deleted = trace.get("deleted", [])
                        if not deleted:
                            return ends_list
                        deleted_sorted = sorted(deleted)
                        new_ends = []
                        for e in ends_list:
                            shift = 0
                            for d in deleted_sorted:
                                if d < e:
                                    shift += 1
                                else:
                                    break
                            new_ends.append(e - shift)
                        return new_ends
                    if op == "insert":
                        j = trace.get("index", None)
                        if j is None:
                            return ends_list
                        return [e + 1 if e >= j else e for e in ends_list]
                    return ends_list

                def _spans_from_ends(keys, ends_list):
                    spans = {}
                    start = 0
                    for k, end in zip(keys, ends_list):
                        spans[k] = (start, end)
                        start = end
                    return spans

                u = Morph(expected_ur)
                for r in global_rules:
                    spans_now = _spans_from_ends(keys_in_word, ends)
                    boundary_positions = set(ends[:-1]) if ends else None
                    u, trace = apply_rule(
                        r,
                        u,
                        bank=self.bank,
                        spans_override=spans_now,
                        boundary_positions_override=boundary_positions,
                        order=order,
                        word_to_slot_values=self.word_to_slot_values,
                        wid=word,
                        return_trace=True,
                    )
                    ends = _update_ends(ends, trace)
                for (slot, val), r in local_rules:
                    if slot in vals and vals[slot] == val:
                        key = (slot, val)
                        spans_now = _spans_from_ends(keys_in_word, ends)
                        boundary_positions = set(ends[:-1]) if ends else None
                        u, trace = apply_rule(
                            r,
                            u,
                            bank=self.bank,
                            allowed_morpheme=key,
                            spans_override=spans_now,
                            boundary_positions_override=boundary_positions,
                            order=order,
                            word_to_slot_values=self.word_to_slot_values,
                            wid=word,
                            return_trace=True,
                        )
                        ends = _update_ends(ends, trace)
                expected_sr = "".join(u.phonemes)
            except Exception:
                traceback.print_exc()
                expected_sr = expected_ur
        else:
            expected_sr = expected_ur
        return expected_sr

    def _generate_candidate_rules(
        self,
        words,
        order,
        segmentation,
        morphemes,
        global_rules,
        local_rules,
    ):
        """
        For a given intermediate theory, find up to self.k best rules of each type (global
        and local for each morpheme) to extend it. Rules are evaluated by (1) how much they
        reduce TFED to observed SRs, and (2) their cost.
        """
        current_cost = self._calculate_total_rule_cost(global_rules, local_rules)
        remaining_cost = self.max_total_cost - current_cost
        if remaining_cost <= 0:
            return []  # No budget for new rules

        # Build URs for all words
        examples = []
        for w in segmentation.keys():
            vals = self.word_to_slot_values.get(w, {})
            breaks = segmentation.get(w, [])
            segs = self.segmenter.get_segments(w, order, breaks) if breaks else {}
            ur_parts = []
            for s in order:
                if s in vals:
                    key = (s, vals[s])
                    if key in morphemes:
                        part = morphemes[key]
                    else:
                        part = segs.get(s, "")
                    ur_parts.append(part)
                else:
                    ur_parts.append("")
            ur = "".join(ur_parts)
            if ur:
                examples.append((ur, w[0], w))

        all_rule_candidates = []

        # Generate global rule candidates
        try:
            if len(examples) > 0:
                examples_for_rs = [
                    (Morph(ur), Morph(sr), wid) for (ur, sr, wid) in examples
                ]
                if self.DEBUG:
                    print("Generating global rules...")
                candidates = RuleSolver(
                    examples_for_rs,
                    self.bank,
                    allowed_morpheme=None,
                    order=order,
                    segmentation=segmentation,
                    morphemes=morphemes,
                    word_to_slot_values=self.word_to_slot_values,
                    pre_global_rules=global_rules,
                    pre_local_rules=local_rules,
                    debug=self.DEBUG,
                ).topK(
                    k=self.k,
                    max_cost=min(self.max_rule_cost, remaining_cost),
                )
                for new_rule in candidates:
                    if float(new_rule.cost()) <= remaining_cost:
                        all_rule_candidates.append(("global", None, new_rule))
                        if self.DEBUG:
                            print(
                                f"Found global rule: {str(new_rule)}, cost={new_rule.cost()}",
                            )
        except Exception:
            traceback.print_exc()

        # Generate local rule candidates for each morpheme
        morpheme_keys_to_consider = set()
        for w in words:
            vals_w = self.word_to_slot_values.get(w, {})
            for slot in order:
                if slot in vals_w:
                    morpheme_keys_to_consider.add((slot, vals_w[slot]))
        for morpheme_key in morpheme_keys_to_consider:
            try:
                if len(examples) > 0:
                    examples_for_rs = [
                        (Morph(ur), Morph(sr), wid) for (ur, sr, wid) in examples
                    ]
                    if self.DEBUG:
                        print(
                            f"Generating local rules for {morpheme_key}..."
                        )
                    candidates = RuleSolver(
                        examples_for_rs,
                        self.bank,
                        allowed_morpheme=morpheme_key,
                        order=order,
                        segmentation=segmentation,
                        morphemes=morphemes,
                        word_to_slot_values=self.word_to_slot_values,
                        pre_global_rules=global_rules,
                        pre_local_rules=local_rules,
                        debug=self.DEBUG,
                    ).topK(
                        k=self.k,
                        max_cost=min(self.max_rule_cost, remaining_cost),
                    )
                    for new_rule in candidates:
                        if float(new_rule.cost()) <= remaining_cost:
                            all_rule_candidates.append(
                                ("local", morpheme_key, new_rule)
                            )
                            if self.DEBUG:
                                print(
                                    f"Found local rule for {morpheme_key}: {str(new_rule)}, cost={new_rule.cost()}",
                                )
            except Exception:
                traceback.print_exc()

        return all_rule_candidates

    def _candidate_segs_from_undo_alignment(
        self,
        key,
        new_morphemes,
        keys_i,
        ur_i: Morph,
        undone_i: Morph,
    ):
        """Locate the given morpheme (`key`) in the given undone UR."""
        old_ur_tokens = list(ur_i.phonemes)
        undone_tokens = list(undone_i.phonemes)

        # Locate the span for this morpheme within the old UR
        span_start, span_end = None, None
        pos = 0
        for key_s in keys_i:
            part_s = new_morphemes.get(key_s, "")
            part_len = len(tokenize(part_s)) if part_s else 0
            start_s, end_s = pos, pos + part_len
            pos = end_s
            if key_s == key:
                span_start, span_end = start_s, end_s
                break

        if span_start is None or span_end is None:
            return []

        # Propose plausible old -> undone mappings
        candidates = set()
        alignments = _align_ops(old_ur_tokens, undone_tokens, bank=self.bank)
        for ops in alignments:
            j_idxs = []
            for op, _u_tok, _s_tok, i_idx, j_idx in ops:
                if op in ("M", "S"):
                    if span_start <= i_idx < span_end:
                        j_idxs.append(j_idx)
                elif op == "D":
                    continue
                elif op == "I":
                    if span_start <= i_idx < span_end:
                        j_idxs.append(j_idx)
                elif op == "T":
                    i0, j0 = i_idx, j_idx
                    if span_start <= i0 < span_end:
                        j_idxs.append(j0)
                    if span_start <= (i0 + 1) < span_end:
                        j_idxs.append(j0 + 1)

            if not j_idxs:
                candidates.add("")
            else:
                j0, j1 = min(j_idxs), max(j_idxs)
                candidates.add("".join(undone_tokens[j0 : j1 + 1]))

        return sorted(candidates, key=lambda s: (len(s), s))

    def _tfed_for_theory(
        self, words, precomputed, order, morphemes, global_rules, local_rules
    ):
        """Compute TFED and list of inconsistent words under the current theory."""
        def _tokens(s: str):
            # Prefer Morph tokenization, which normalizes certain characters (e.g., ä -> æ).
            try:
                return Morph(s).phonemes
            except Exception:
                return list(tokenize(s))

        t = 0.0
        inconsistent_words = []
        for ww in words:
            surface, vals, _segs = precomputed[ww]
            pred = self._predict_sr_for_word(
                ww, vals, order, morphemes, global_rules, local_rules
            )
            d = float(self.ed.edit_distance(pred, surface))
            t += d
            if _tokens(pred) != _tokens(surface):
                inconsistent_words.append(ww)
        return t, inconsistent_words

    def _repair_lexicon(
        self,
        words,
        precomputed,
        order,
        morphemes,
        global_rules,
        local_rules,
        word_keys_in_order,
        max_iters=50,
    ):
        """
        Check if a rule is really redundant by checking if all UR -> SR changes are explained
        by other rules. This will require revising URs for words that are not yet consistent
        (we do so by inverting remaining rules).
        """
        new_morphemes = dict(morphemes)

        # Candidate rules we can safely invert on SR strings: for now, these are invertible
        # global rules that do not mention '+'. In principle, it is possible to invert
        # local rules or those with '+' in the guards, but this would require more careful
        # handling (TODO: implement).
        invertible_rules = []
        for r in list(global_rules):
            if not can_invert_rule(r):
                continue
            if any(
                isinstance(s, BoundarySpecification)
                for s in r.leftTriggers.specifications + r.rightTriggers.specifications
            ):
                continue
            inv = invert_rule(r)
            setattr(inv, "locality_mode", getattr(r, "locality_mode", None))
            invertible_rules.append(inv)

        if not invertible_rules:
            return new_morphemes

        for _iter in range(int(max_iters)):
            improved = False

            # Identify inconsistent words under the current theory
            tfed, inconsistent = self._tfed_for_theory(
                words, precomputed, order, new_morphemes, global_rules, local_rules
            )
            if not inconsistent or tfed <= 0.0:
                break

            # For each inconsistent word, try inverting one invertible rule on its SR to get
            # the new UR
            for wid in list(inconsistent):
                surface, vals, _segs = precomputed[wid]
                keys_i = word_keys_in_order.get(wid, [])
                if not keys_i:
                    continue

                ur_i = "".join(new_morphemes.get(k, "") for k in keys_i)
                ur_i_m = Morph(ur_i)

                sr_m = Morph(surface)
                for inv_rule in invertible_rules:
                    try:
                        undone_i = apply_rule(
                            inv_rule,
                            sr_m,
                            bank=self.bank,
                            allowed_morpheme=None,
                            morphemes=new_morphemes,
                            order=order,
                            word_to_slot_values=self.word_to_slot_values,
                            wid=wid,
                        )
                    except Exception:
                        continue

                    # Use alignment (ur_i -> undone_i) to propose updated UR strings for
                    # morphemes in this word
                    for key in keys_i:
                        for candidate_seg in self._candidate_segs_from_undo_alignment(
                            key, new_morphemes, keys_i, ur_i_m, undone_i
                        ):
                            if candidate_seg == new_morphemes.get(key, ""):
                                continue
                            candidate_lexicon = dict(new_morphemes)
                            candidate_lexicon[key] = candidate_seg
                            cand_tfed, _cand_bad = self._tfed_for_theory(
                                words,
                                precomputed,
                                order,
                                candidate_lexicon,
                                global_rules,
                                local_rules,
                            )

                            # Keep any morpheme update that strictly lowers TFED
                            if cand_tfed < tfed:
                                if self.DEBUG:
                                    print(
                                        f"[prune] Lexicon repair improved TFED {tfed} -> {cand_tfed} "
                                        f"by setting {key}={candidate_seg!r}",
                                    )
                                new_morphemes = candidate_lexicon
                                tfed = cand_tfed
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return new_morphemes

    def _prune_redundant_rules(
        self,
        words,
        precomputed,
        order,
        morphemes,
        global_rules,
        local_rules,
    ):
        """Post-solve simplification pass: find if any rules in the solution are redundant."""
        if self.DEBUG:
            print(
                f"[prune] Starting redundant-rule pruning: "
                f"{len(global_rules)} global, {len(local_rules)} local rules",
            )

        word_keys_in_order = {}
        for w in words:
            vals = self.word_to_slot_values.get(w, {})
            word_keys_in_order[w] = [(s, vals[s]) for s in order if s in vals]

        m = dict(morphemes)
        g = list(global_rules)
        l = list(local_rules)

        # Prioritize removing expensive rules first
        def _candidates():
            cands = []
            for i, rr in enumerate(g):
                cands.append(("global", i, None, rr, float(rr.cost())))
            for i, (k, rr) in enumerate(l):
                cands.append(("local", i, k, rr, float(rr.cost())))
            cands.sort(key=lambda t: (-t[4], t[0], t[1]))
            return cands

        made_change = True
        while made_change:
            made_change = False
            # For each rule r (local or global)
            for kind, idx, key, rr, _cost in _candidates():
                g2 = list(g)
                l2 = list(l)

                # Try to remove r
                if kind == "global":
                    if idx < 0 or idx >= len(g2):
                        continue
                    removed = g2.pop(idx)
                else:
                    if idx < 0 or idx >= len(l2):
                        continue
                    removed = l2.pop(idx)
                    removed = removed[1]

                # If TFED stays 0, keep it removed
                tfed2, bad2 = self._tfed_for_theory(
                    words, precomputed, order, m, g2, l2
                )
                if not bad2:
                    if self.DEBUG:
                        print(
                            f"[prune] Removed redundant rule (no lexicon change): {removed}",
                        )
                    g, l = g2, l2
                    made_change = True
                    break

                # Otherwise, attempt to repair the lexicon using inverses of remaining
                # (safe) rules
                m2 = self._repair_lexicon(
                    words,
                    precomputed,
                    order,
                    m,
                    g2,
                    l2,
                    word_keys_in_order,
                )

                # If TFED returns to 0, keep r removed and keep the repaired lexicon
                tfed3, bad3 = self._tfed_for_theory(
                    words, precomputed, order, m2, g2, l2
                )
                if not bad3:
                    if self.DEBUG:
                        print(
                            f"[prune] Removed rule after lexicon repair: {removed}",
                        )
                        print(
                            f"[prune] Rule cost { self._calculate_total_rule_cost(g, l) } -> { self._calculate_total_rule_cost(g2, l2) }",
                        )
                    m, g, l = m2, g2, l2
                    made_change = True
                    break

        if self.DEBUG:
            print(
                f"[prune] Done redundant-rule pruning: "
                f"{len(g)} global, {len(l)} local rules",
            )
        return m, g, l

    def _find_minimal_theory(self, order, segmentation):
        """Find a theory (morpheme URs and rules) that makes the segmentation consistent."""
        # Pre-compute metadata for all words
        precomputed = {}
        for w in self.words:
            surface = w[0]
            vals = self.word_to_slot_values.get(w, {})
            breaks = segmentation[w]
            segs = self.segmenter.get_segments(w, order, breaks)
            precomputed[w] = (surface, vals, segs)

        word_keys_in_order = {}
        for w in self.words:
            vals = self.word_to_slot_values.get(w, {})
            keys = [(s, vals[s]) for s in order if s in vals]
            word_keys_in_order[w] = keys

        # Build candidate URs for each morpheme key by collecting all observed
        # surface realizations of that morpheme under this segmentation.
        key_to_seg_counter = defaultdict(Counter)
        for w in self.words:
            _surface, vals, segs = precomputed[w]
            for slot in order:
                if slot not in vals:
                    continue
                key = (slot, vals[slot])
                key_to_seg_counter[key][segs.get(slot, "")] += 1

        # Deterministic ordering: most frequent first, then shorter, then lexicographic
        key_to_candidates = {}
        for key, ctr in key_to_seg_counter.items():
            cands = sorted(
                ctr.items(),
                key=lambda kv: (-kv[1], len(kv[0]), kv[0]),
            )
            key_to_candidates[key] = [seg for (seg, _count) in cands]
            if not key_to_candidates[key]:
                key_to_candidates[key] = [""]

        # Priority queue over theories:
        # (current_tfed, total_cost, counter, morphemes, global_rules, local_rules)
        queue = []
        counter = 0
        visited = set()

        # Build initial lexica as the full Cartesian product of per-morpheme
        # candidates. This tries every observed SR segment as the UR for each
        # morpheme, in every combination with other morphemes' UR choices.
        slot_rank = {slot: i for i, slot in enumerate(order)}
        keys_sorted = sorted(
            key_to_candidates.keys(),
            key=lambda kv: (slot_rank.get(kv[0], 10**9), str(kv[1])),
        )
        candidate_lists = [key_to_candidates[k] for k in keys_sorted]

        initial_states_pushed = 0

        for combo in (
            itertools.product(*candidate_lists) if candidate_lists else [tuple()]
        ):
            m = {k: seg for k, seg in zip(keys_sorted, combo)}
            initial_tfed, _ = self._tfed_for_theory(
                self.words, precomputed, order, m, [], {}
            )
            heapq.heappush(queue, (initial_tfed, 0.0, counter, m, [], []))
            counter += 1
            initial_states_pushed += 1
            if initial_states_pushed >= self.initial_lexicon_max_states:
                break

        if self.DEBUG:
            print(
                f"Initial lexica: {initial_states_pushed} states",
            )

        while queue:
            current_tfed, current_cost, _, morphemes, global_rules, local_rules = (
                heapq.heappop(queue)
            )

            # Respect the cost budget
            if current_cost > self.max_total_cost:
                continue

            # State memoization (rules + morpheme lexicon)
            state_key = (
                tuple(sorted(global_rules)),
                tuple(sorted(local_rules)),
                None if morphemes is None else frozenset(morphemes.items()),
            )
            if state_key in visited:
                continue
            visited.add(state_key)

            # Check consistency for all words under the current theory
            tfed, inconsistent_words = self._tfed_for_theory(
                self.words, precomputed, order, morphemes, global_rules, local_rules
            )

            # If all words are consistent, check for redundant rules and return
            if not inconsistent_words:
                try:
                    morphemes2, global_rules2, local_rules2 = (
                        self._prune_redundant_rules(
                            self.words,
                            precomputed,
                            order,
                            morphemes,
                            global_rules,
                            local_rules,
                        )
                    )
                    cost2 = self._calculate_total_rule_cost(global_rules2, local_rules2)
                    return cost2, morphemes2, global_rules2, local_rules2
                except Exception:
                    traceback.print_exc()
                    return current_cost, morphemes, global_rules, local_rules

            if self.DEBUG:
                print(
                    f"Current popped state: TFED={tfed}, cost={current_cost}, morphemes={morphemes}, global_rules={global_rules}, local_rules={local_rules}",
                )

            # Explore extensions of the current theory by one additional rule
            all_rule_options = self._generate_candidate_rules(
                inconsistent_words,
                order,
                segmentation,
                morphemes,
                global_rules,
                local_rules,
            )

            next_states = []
            min_new_tfed = float("inf")

            for rule_type, morpheme_key, new_rule in all_rule_options:
                new_global_rules = global_rules.copy()
                new_local_rules = local_rules.copy()

                if rule_type == "global":
                    new_global_rules.append(new_rule)
                else:
                    if (morpheme_key, new_rule) not in new_local_rules:
                        new_local_rules.append((morpheme_key, new_rule))

                new_cost = self._calculate_total_rule_cost(
                    new_global_rules, new_local_rules
                )
                if new_cost > self.max_total_cost:
                    continue

                # Re-evaluate TFED under the proposed theory
                new_tfed, _ = self._tfed_for_theory(
                    self.words,
                    precomputed,
                    order,
                    morphemes,
                    new_global_rules,
                    new_local_rules,
                )

                # Hypothesis: some URs are themselves rule outputs.
                # Try to "undo" the rule for each morpheme and word
                # and keep a change only if it reduces TFED.
                new_morphemes = dict(morphemes)
                if new_tfed > 0 and can_invert_rule(new_rule):
                    inverse_rule = invert_rule(new_rule)
                    setattr(inverse_rule, "locality_mode", new_rule.locality_mode)
                    if self.DEBUG:
                        print(
                            f"Inverse rule: {str(inverse_rule)}, allowed_morpheme={morpheme_key}, locality_mode={inverse_rule.locality_mode}"
                        )

                    # For each morpheme, find and keep the best UR among the
                    # current UR and the results of undoing the rule in all
                    # words containing the morpheme.
                    for key in new_morphemes:
                        best_tfed = new_tfed
                        best_seg = new_morphemes[key]
                        slot_k, val_k = key

                        # Find all words containing the morpheme
                        occurrences = self.slot_to_value_to_words[slot_k][val_k]

                        # Derive the "undone" UR for each word
                        for wid_i in occurrences:
                            ur_i = "".join(
                                new_morphemes[k] for k in word_keys_in_order[wid_i]
                            )
                            ur_i = Morph(ur_i)
                            try:
                                undone_i = apply_rule(
                                    inverse_rule,
                                    ur_i,
                                    bank=self.bank,
                                    allowed_morpheme=morpheme_key,
                                    morphemes=new_morphemes,
                                    order=order,
                                    word_to_slot_values=self.word_to_slot_values,
                                    wid=wid_i,
                                )
                            except Exception:
                                continue

                            if self.DEBUG:
                                print(f"ur_i={str(ur_i)}, undone_i={str(undone_i)}")

                            # Find the "undone" UR in the word
                            keys_i = word_keys_in_order[wid_i]
                            for (
                                candidate_seg
                            ) in self._candidate_segs_from_undo_alignment(
                                key, new_morphemes, keys_i, ur_i, undone_i
                            ):
                                candidate_lexicon = dict(new_morphemes)
                                candidate_lexicon[key] = candidate_seg
                                candidate_tfed, _ = self._tfed_for_theory(
                                    self.words,
                                    precomputed,
                                    order,
                                    candidate_lexicon,
                                    new_global_rules,
                                    new_local_rules,
                                )
                                if self.DEBUG:
                                    print(f"{candidate_tfed=}")
                                if candidate_tfed < best_tfed:
                                    best_tfed = candidate_tfed
                                    best_seg = candidate_seg

                            if best_seg != new_morphemes[key]:
                                new_morphemes[key] = best_seg
                                new_tfed = best_tfed
                                if self.DEBUG:
                                    print(
                                        f"Revised key {key} to {best_seg}, new TFED: {best_tfed}, morphemes: {new_morphemes}",
                                    )

                if self.DEBUG:
                    print(
                        f"New TFED: {new_tfed}, cost: {new_cost}, morphemes: {new_morphemes}, global_rules: {new_global_rules}, local_rules: {new_local_rules}",
                    )

                # Require that the new theory strictly decreases TFED
                if new_tfed >= tfed:
                    continue

                new_state_key = (
                    tuple(sorted(new_global_rules)),
                    tuple(sorted(new_local_rules)),
                    None if new_morphemes is None else frozenset(new_morphemes.items()),
                )
                if new_state_key in visited:
                    continue

                next_state = (
                    new_tfed,
                    new_cost,
                    counter,
                    new_morphemes,
                    new_global_rules,
                    new_local_rules,
                )

                next_states.append(next_state)
                min_new_tfed = min(min_new_tfed, new_tfed)

            for next_state in next_states:
                if next_state[0] == min_new_tfed:
                    if self.DEBUG:
                        print(f"Pushing next state: {next_state}")
                    counter += 1
                    heapq.heappush(queue, next_state)

        return None

    def _calculate_total_rule_cost(self, global_rules, local_rules):
        """Calculate total cost of all rules."""
        total = 0.0
        for rule in global_rules:
            total += float(rule.cost())
        for _, rule in local_rules:
            total += float(rule.cost())
        return total

    def _solve_with_order(self, order):
        """Search for morphemes and phonological rules given a fixed morpheme order."""
        # Generate segmentations and keep only those with lowest TFED
        all_segmentations = self.segmenter.generate_segmentations(order)

        for segmentation, tfed in all_segmentations:
            if self.DEBUG:
                print(f"Trying segmentation (TFED={tfed}): {segmentation}")

            theory = self._find_minimal_theory(order, segmentation)

            if theory is None:
                # No theory within cost budget fits segmentation
                continue

            total_cost, morphemes, global_rules, local_rules = theory
            lexicon = [(seg, v) for (_, v), seg in morphemes.items() if seg]
            global_rules = set(global_rules)
            local_rules = set(local_rules)
            return list(order), sorted(set(lexicon)), global_rules, local_rules

        # Unsatisfiable for this order
        return None

    def solve(self):
        """Search for morphemes and phonological rules."""
        # Estimate which orders are most likely to be correct; try them first.
        orders = list(itertools.permutations(self.slots))
        scored = [
            (self.order_scorer.score_order(list(order)), list(order))
            for order in orders
        ]
        scored.sort(key=lambda x: -x[0])  # Sort descending by score

        for score, order in scored:
            if self.DEBUG:
                print(f"Trying order (score={score:.2f}): {order}")
            res = self._solve_with_order(order)
            if res is not None:
                return res

        return None
