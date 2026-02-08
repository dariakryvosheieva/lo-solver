import traceback, os, time
from fractions import Fraction
from collections import defaultdict
from contextlib import contextmanager

from rule import (
    Rule,
    ConstantPhoneme,
    FeatureMatrix,
    EmptySpecification,
    OffsetSpecification,
    BoundarySpecification,
    PlaceSpecification,
    Guard,
    MetathesisFocus,
    MetathesisSpecification,
)
from morph import Morph
from features import tokenize
from rule_application import apply_rule, matches_specification
from edit_distance_calculator import EditDistanceCalculator


def _subst_cost(p, q, bank):
    """
    Normalized cost of substituting p for q:
        (# of differing features) / (total # of features in the bank).
    """
    try:
        total = len(bank.features)
        vp = bank.featureVectorMap.get(p, [])
        vq = bank.featureVectorMap.get(q, [])
        diff = int(sum(1 for x, y in zip(vp, vq, strict=True) if bool(x) != bool(y)))
        return Fraction(diff, total)
    except Exception:
        traceback.print_exc()
        return Fraction(1, 1)


def _align_ops(ur_tokens, sr_tokens, bank):
    """
    Feature-aware edit alignment over phoneme tokens.

    Returns a list of minimal-cost alignments.
    Each alignment is a list of ops, and each op is a tuple:
        ("M"|"S"|"I"|"D", ur_token, sr_token, i_index, j_index).
    - "M": match, "S": substitute, "I": insert, "D": delete
    - ur_token: the UR token matched, substituted, or deleted (None for insertions)
    - sr_token: the SR token matched, substituted, or inserted (None for deletions)
    - i_index: index of ur_token in the UR token list
    - j_index: index of sr_token in the SR token list

    Cost model:
    - Matches have cost 0
    - Insertions and deletions have fixed cost 1
    - Substitutions have cost (# differing features) / (# features in the bank)
    - Metathesis/transposition (swap of two adjacent *distinct* tokens)
      has a very small positive cost (epsilon), and is only allowed when
      the two tokens are swapped exactly. (This is used for observation
      extraction, not scoring.)
    """
    m, n = len(ur_tokens), len(sr_tokens)
    dp = [[Fraction(0, 1) for _ in range(n + 1)] for _ in range(m + 1)]
    back = [[[] for _ in range(n + 1)] for _ in range(m + 1)]

    indel_cost = Fraction(1, 1)

    for i in range(1, m + 1):
        # Deleting one UR token
        dp[i][0] = dp[i - 1][0] + indel_cost
        back[i][0] = ["D"]
    for j in range(1, n + 1):
        # Inserting one SR token
        dp[0][j] = dp[0][j - 1] + indel_cost
        back[0][j] = ["I"]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            del_c = dp[i - 1][j] + indel_cost
            ins_c = dp[i][j - 1] + indel_cost

            sub_cost = _subst_cost(ur_tokens[i - 1], sr_tokens[j - 1], bank)
            sub_c = dp[i - 1][j - 1] + sub_cost

            # Adjacent transposition (metathesis)
            trans_c = None
            if i >= 2 and j >= 2:
                a0, a1 = ur_tokens[i - 2], ur_tokens[i - 1]
                b0, b1 = sr_tokens[j - 2], sr_tokens[j - 1]
                if a0 != a1 and a0 == b1 and a1 == b0:
                    denom = max(1, len(getattr(bank, "features", [])) * 10)
                    trans_cost = Fraction(1, denom)
                    trans_c = dp[i - 2][j - 2] + trans_cost

            candidates_costs = [del_c, ins_c, sub_c]
            if trans_c is not None:
                candidates_costs.append(trans_c)

            best = min(candidates_costs)
            dp[i][j] = best

            # Collect all minimal-cost backpointers
            candidates = []
            candidates.append(
                ("S" if ur_tokens[i - 1] != sr_tokens[j - 1] else "M", sub_c)
            )
            candidates.append(("I", ins_c))
            candidates.append(("D", del_c))
            if trans_c is not None:
                candidates.append(("T", trans_c))
            candidates = [c for c in candidates if c[1] == best]

            op_idx = {"M": 0, "S": 1, "T": 2, "I": 3, "D": 3}
            back[i][j] = [
                op for op, _c in sorted(candidates, key=lambda t: (op_idx[t[0]], t[0]))
            ]

    # Enumerate all minimal-cost alignments via backpointers
    memo = {}

    def _build(i, j):
        key = (i, j)
        if key in memo:
            return memo[key]
        if i == 0 and j == 0:
            memo[key] = [[]]
            return memo[key]

        out = []
        for op in back[i][j]:
            if op == "M" or op == "S":
                prev = _build(i - 1, j - 1)
                step = (op, ur_tokens[i - 1], sr_tokens[j - 1], i - 1, j - 1)
            elif op == "T":
                prev = _build(i - 2, j - 2)
                step = (
                    op,
                    (ur_tokens[i - 2], ur_tokens[i - 1]),
                    (sr_tokens[j - 2], sr_tokens[j - 1]),
                    i - 2,
                    j - 2,
                )
            elif op == "I":
                prev = _build(i, j - 1)
                step = (op, None, sr_tokens[j - 1], i, j - 1)
            elif op == "D":
                prev = _build(i - 1, j)
                step = (op, ur_tokens[i - 1], None, i - 1, j)
            else:
                continue
            for seq in prev:
                out.append(seq + [step])

        memo[key] = out
        return out

    return _build(m, n)


def can_invert_rule(r):
    """Decide whether we can try to invert a rule."""
    # The following rules can only be inverted
    # deterministically if the focus is a ConstantPhoneme:
    # - deletion
    # - substitution with ConstantPhoneme change
    # - substitution with OffsetSpecification change
    if isinstance(r.structuralChange, EmptySpecification):
        return isinstance(r.focus, ConstantPhoneme)
    if isinstance(r.structuralChange, ConstantPhoneme) and not isinstance(
        r.focus, EmptySpecification
    ):
        return isinstance(r.focus, ConstantPhoneme)
    if isinstance(r.structuralChange, OffsetSpecification) and not isinstance(
        r.focus, EmptySpecification
    ):
        return isinstance(r.focus, ConstantPhoneme)
    # Other rules are always invertible
    return True


def _flip_feature_matrix(feature_matrix):
    """Flip the polarities of a feature matrix."""
    return FeatureMatrix(
        [(not v, f) for (v, f) in feature_matrix.featuresAndPolarities]
    )


def invert_rule(rule):
    """Construct the inverse of a rule."""
    # Preserve the guards
    left_guard = rule.leftTriggers
    right_guard = rule.rightTriggers

    # The following rules can be undone by swapping change and focus:
    # - insertion
    # - deletion (ConstantPhoneme focus assumed)
    # - substitution with ConstantPhoneme or OffsetSpecification change
    #   (ConstantPhoneme focus assumed)
    # - substitution with FeatureMatrix change where focus and change have different
    #   polarities for the same feature
    if (
        isinstance(rule.focus, EmptySpecification)
        or (
            isinstance(rule.focus, ConstantPhoneme)
            and isinstance(
                rule.structuralChange,
                (EmptySpecification, ConstantPhoneme, OffsetSpecification),
            )
        )
        or (
            isinstance(rule.structuralChange, FeatureMatrix)
            and isinstance(rule.focus, FeatureMatrix)
            and rule.structuralChange == _flip_feature_matrix(rule.focus)
        )
    ):
        return Rule(rule.structuralChange, rule.focus, left_guard, right_guard)

    # Substitution with FeatureMatrix change:
    # undo by flipping the feature
    if isinstance(rule.structuralChange, FeatureMatrix):
        return Rule(
            rule.focus,
            FeatureMatrix(
                [(not v, f) for (v, f) in rule.structuralChange.featuresAndPolarities]
            ),
            left_guard,
            right_guard,
        )

    # Metathesis: invert by swapping the ordered focus
    if isinstance(rule.focus, MetathesisFocus) and isinstance(
        rule.structuralChange, MetathesisSpecification
    ):
        return Rule(
            MetathesisFocus(rule.focus.s2, rule.focus.s1),
            rule.structuralChange,
            left_guard,
            right_guard,
        )


class RuntimeProfiler:
    """Track the runtime of different parts of the rule search algorithm."""

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.totals_s = defaultdict(float)  # key -> seconds
        self.counts = defaultdict(int)  # key -> count

    def reset(self):
        self.totals_s.clear()
        self.counts.clear()

    def inc(self, key, n=1):
        if not self.enabled:
            return
        self.counts[key] += int(n)

    def add_time(self, key, dt_s, n=1):
        if not self.enabled:
            return
        self.totals_s[key] += float(dt_s)
        self.counts[key] += int(n)

    @contextmanager
    def span(self, key, n=1):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(key, time.perf_counter() - t0, n=n)

    def report(self, header="RuntimeProfiler report", top_n=30, min_total_ms=0.0):
        if not self.enabled:
            return
        items = []
        for k, total_s in self.totals_s.items():
            total_ms = total_s * 1e3
            if total_ms < float(min_total_ms):
                continue
            items.append((total_s, k))
        items.sort(reverse=True)

        print(f"{header}")
        for total_s, k in items[: int(top_n)]:
            c = self.counts.get(k, 0)
            per_ms = (total_s * 1e3 / c) if c else 0.0
            print(
                f"  {k:40s}  total={total_s:9.3f}s  count={c:7d}  avg={per_ms:9.3f}ms",
            )


class Observation:
    """Representation for UR/SR inconsistencies that need to be explained by rules."""

    def __init__(self, id, X, Y, L, R):
        self.id = id
        self.context = (X, Y, L, R)
        self.morpheme_boundary_data = {
            "L": {
                "boundary_only": False,
                "spec_plus_boundary": set(),
            },
            "R": {
                "boundary_only": False,
                "boundary_plus_spec": set(),
            },
        }


class RuleSolver:
    """Constraint-based solver for phonological rules."""

    def __init__(
        self,
        examples=[],
        bank=None,
        allowed_morpheme=None,
        order=None,
        segmentation=None,
        morphemes=None,
        word_to_slot_values={},
        pre_global_rules=[],
        pre_local_rules=[],
        debug=True,
    ):
        self._input_examples = []
        self._input_ur_by_wid = {}
        for ur, sr, wid in examples:
            self._input_examples.append((ur, sr, wid))
            self._input_ur_by_wid[wid] = ur

        self.bank = bank
        self.allowed_morpheme = allowed_morpheme
        self.order = order
        self.segmentation = segmentation
        self.morphemes = morphemes
        self.word_to_slot_values = word_to_slot_values
        self.pre_global_rules = pre_global_rules
        self.pre_local_rules = pre_local_rules
        self.DEBUG = debug

        self._prof = RuntimeProfiler(enabled=debug)

        # Precompute morphology lookups used in TFED evaluation
        self._word_keys_in_order = {}
        self._morpheme_to_wids = defaultdict(tuple)
        for _ur, sr, wid in self._input_examples:
            vals = self.word_to_slot_values.get(wid, {})
            self._word_keys_in_order[wid] = tuple(
                (slot, vals[slot]) for slot in self.order if slot in vals
            )
            for k in self._word_keys_in_order[wid]:
                self._morpheme_to_wids[k] += (wid,)

        # Tokenization cache to avoid repeated O(len(word)^2) work in TFED evaluation
        self._seg_tokens_cache = {}
        self._phoneme_len_cache = {}

        # Precompute effective (intermediate) URs after applying existing rules.
        # These are the forms that new rules should be applied to.
        self._effective_boundary_starts_by_wid = {}
        self._effective_boundary_positions_by_wid = {}
        self._effective_spans_by_wid = {}
        self.all_examples = []
        for _ur_in, sr, wid in self._input_examples:
            ur_eff, spans_eff, boundary_positions_eff = self._effective_ur_meta_for_wid(
                wid, self.morphemes
            )
            self._effective_spans_by_wid[wid] = spans_eff
            self._effective_boundary_positions_by_wid[wid] = set(
                boundary_positions_eff or []
            )
            self._effective_boundary_starts_by_wid[wid] = set(
                boundary_positions_eff or []
            )
            self.all_examples.append((ur_eff, sr, wid))

        self.observations = self._extract_observations()
        if self.DEBUG:
            print(
                f"Observations: {[(obs.id, obs.context, obs.morpheme_boundary_data) for obs in self.observations]}",
            )

    def _apply_rules_with_boundary_tracking(
        self,
        wid,
        ur_morph,
        morpheme_lex,
        global_rules,
        local_rules,
    ):
        """Apply a sequence of rules while tracking changes to morpheme boundaries."""
        vals = self.word_to_slot_values.get(wid, {})
        keys_in_word = self._word_keys_in_order.get(wid, ())

        ends = []
        pos = 0
        for key in keys_in_word:
            part = morpheme_lex.get(key, "")
            pos += self._phoneme_len(part)
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

        u = ur_morph
        for r in global_rules:
            spans_now = _spans_from_ends(keys_in_word, ends)
            boundary_positions = set(ends[:-1]) if ends else None
            u, trace = apply_rule(
                r,
                u,
                bank=self.bank,
                spans_override=spans_now,
                boundary_positions_override=boundary_positions,
                morphemes=morpheme_lex,
                order=self.order,
                word_to_slot_values=self.word_to_slot_values,
                wid=wid,
                return_trace=True,
            )
            ends = _update_ends(ends, trace)

        for (slot, val), r in local_rules:
            if vals.get(slot) != val:
                continue
            spans_now = _spans_from_ends(keys_in_word, ends)
            boundary_positions = set(ends[:-1]) if ends else None
            u, trace = apply_rule(
                r,
                u,
                bank=self.bank,
                allowed_morpheme=(slot, val),
                spans_override=spans_now,
                boundary_positions_override=boundary_positions,
                morphemes=morpheme_lex,
                order=self.order,
                word_to_slot_values=self.word_to_slot_values,
                wid=wid,
                return_trace=True,
            )
            ends = _update_ends(ends, trace)

        spans_final = _spans_from_ends(keys_in_word, ends)
        boundary_positions_final = set(ends[:-1]) if ends else None
        return u, spans_final, boundary_positions_final

    def _base_ur_meta_for_wid(self, wid, morpheme_lex):
        """
        Build the deepest UR (lexicon concatenation) and its morpheme spans/boundary
        positions.
        """
        if wid not in self._word_keys_in_order:
            ur_in = self._input_ur_by_wid.get(wid, Morph(""))
            return ur_in, None, None

        keys = self._word_keys_in_order.get(wid, ())

        ur_tokens = []
        ends = []
        pos = 0
        for key in keys:
            seg = morpheme_lex.get(key, "")
            toks = self._seg_tokens(seg)
            ur_tokens.extend(toks)
            pos += len(toks)
            ends.append(pos)
        base_ur = Morph(ur_tokens)

        spans = {}
        start = 0
        for k, end in zip(keys, ends):
            spans[k] = (start, end)
            start = end

        boundary_positions = set(ends[:-1]) if ends else None
        return base_ur, spans, boundary_positions

    def _effective_ur_meta_for_wid(self, wid, morpheme_lex):
        """
        Build the intermediate UR (after existing rules) and its morpheme spans/boundary
        positions.
        """
        if wid not in self._word_keys_in_order:
            ur_in = self._input_ur_by_wid.get(wid, Morph(""))
            return ur_in, None, None

        base_ur, spans, boundary_positions = self._base_ur_meta_for_wid(
            wid, morpheme_lex
        )
        if not self.pre_global_rules and not self.pre_local_rules:
            return base_ur, spans, boundary_positions

        return self._apply_rules_with_boundary_tracking(
            wid,
            base_ur,
            morpheme_lex,
            global_rules=self.pre_global_rules,
            local_rules=self.pre_local_rules,
        )

    def _phoneme_len(self, s):
        """Cached token length (in phonemes) for a segment string."""
        if not s:
            return 0
        cached = self._phoneme_len_cache.get(s)
        if cached is not None:
            return cached
        n = len(self._seg_tokens(s))
        self._phoneme_len_cache[s] = n
        return n

    def _seg_tokens(self, s):
        """Caching wrapper around `tokenize`."""
        if not s:
            return tuple()
        cached = self._seg_tokens_cache.get(s)
        if cached is not None:
            return cached
        toks = tuple(tokenize(s))
        self._seg_tokens_cache[s] = toks
        self._phoneme_len_cache.setdefault(s, len(toks))
        return toks

    def _build_ur_from_keys(self, keys_in_order, morpheme_lex):
        """Build UR string by concatenating morphemes for precomputed keys."""
        return "".join(morpheme_lex.get(k, "") for k in keys_in_order)

    def _extract_observations(self):
        """
        Extract observations of the form (X, Y, L, R)
        meaning "X becomes Y between L and R".

        Each variable is a phoneme or "" (empty string):
        - X == "" for insertions (Ø -> Y)
        - Y == "" for deletions (X -> Ø)
        - L == "" when the change is at word start
        - R == "" when the change is at word end

        Also extract boundary data for guard generation,
        of the form
            {
                "L": {
                    "boundary_only": bool,
                    "spec_plus_boundary": set(phonemes)
                },
                "R": {
                    "boundary_only": bool,
                    "boundary_plus_spec": set(phonemes)
                }
            }
        - "boundary_only" is True if the guard "+" is possible
        - "spec_plus_boundary" is the set of phonemes permitted in guards "spec +"
        - "boundary_plus_spec" is the set of phonemes permitted in guards "+ spec"
        """
        observations = []
        id = 0

        def _slice_local_window_if_needed(
            wid,
            ur_tokens_full,
            sr_tokens_full,
            sr_boundaries_full,
            ur_boundary_starts_full,
        ):
            """
            When learning a local rule, restrict observation extraction to the local
            neighborhood:
                [left neighbor if exists, allowed morpheme, right neighbor if exists].
            Changes may only happen inside this neighborhood, but left and right contexts
            may come from outside.
            """
            if self.allowed_morpheme is None:
                return {
                    "ur_tokens": ur_tokens_full,
                    "sr_tokens": sr_tokens_full,
                    "sr_boundaries": sr_boundaries_full,
                    "ur_boundary_starts": ur_boundary_starts_full,
                    "ur_offset": 0,
                    "sr_offset": 0,
                    "allowed_span_abs": None,
                    "allowed_span_slice": None,
                }

            keys_in_word = self._word_keys_in_order.get(wid, ())
            if not keys_in_word or self.allowed_morpheme not in keys_in_word:
                return {
                    "ur_tokens": ur_tokens_full,
                    "sr_tokens": sr_tokens_full,
                    "sr_boundaries": sr_boundaries_full,
                    "ur_boundary_starts": ur_boundary_starts_full,
                    "ur_offset": 0,
                    "sr_offset": 0,
                    "allowed_span_abs": None,
                    "allowed_span_slice": None,
                }

            k_idx = keys_in_word.index(self.allowed_morpheme)
            left_key = keys_in_word[k_idx - 1] if k_idx > 0 else None
            right_key = (
                keys_in_word[k_idx + 1] if (k_idx + 1) < len(keys_in_word) else None
            )

            # UR window
            ur_tokens = ur_tokens_full
            ur_boundary_starts = ur_boundary_starts_full
            spans_eff = self._effective_spans_by_wid.get(wid)
            allowed_span_abs = None
            allowed_span_slice = None
            local_span_abs = None
            ur_offset = 0
            try:
                if spans_eff and self.allowed_morpheme in spans_eff:
                    a0, a1 = spans_eff[self.allowed_morpheme]
                    a0 = max(0, int(a0))
                    a1 = max(a0, int(a1))
                    allowed_span_abs = (a0, a1)

                    u0 = a0
                    u1 = a1
                    if left_key is not None and left_key in spans_eff:
                        u0 = min(u0, int(spans_eff[left_key][0]))
                    if right_key is not None and right_key in spans_eff:
                        u1 = max(u1, int(spans_eff[right_key][1]))

                    u0 = max(0, int(u0))
                    u1 = max(u0, min(len(ur_tokens_full), int(u1)))
                    ur_tokens = ur_tokens_full[u0:u1]
                    ur_offset = u0
                    allowed_span_slice = (a0 - u0, a1 - u0)
                    local_span_abs = (u0, u1)
                    if ur_boundary_starts is not None:
                        ur_boundary_starts = {
                            (b - u0) for b in set(ur_boundary_starts) if u0 <= b <= u1
                        }
            except Exception:
                # Fall back to the full word
                ur_tokens = ur_tokens_full
                ur_boundary_starts = ur_boundary_starts_full
                allowed_span_abs = None
                allowed_span_slice = None
                local_span_abs = None
                ur_offset = 0

            # SR window
            sr_tokens = sr_tokens_full
            sr_boundaries = sr_boundaries_full
            breaks = self.segmentation.get(wid)
            sr_offset = 0
            if breaks is not None:
                slot_to_idx = {slot: i for i, slot in enumerate(self.order)}

                def _slot_span(slot_name):
                    si = slot_to_idx.get(slot_name)
                    if si is None:
                        return None
                    if si < 0 or si >= len(breaks):
                        return None
                    start = int(breaks[si])
                    end = (
                        int(breaks[si + 1])
                        if (si + 1) < len(breaks)
                        else len(sr_tokens_full)
                    )
                    start = max(0, start)
                    end = max(start, min(len(sr_tokens_full), end))
                    return (start, end)

                allowed_span = _slot_span(self.allowed_morpheme[0])
                if allowed_span is not None:
                    s0, s1 = allowed_span
                    if left_key is not None:
                        left_span = _slot_span(left_key[0])
                        if left_span is not None:
                            s0 = left_span[0]
                    if right_key is not None:
                        right_span = _slot_span(right_key[0])
                        if right_span is not None:
                            s1 = right_span[1]

                    sr_tokens = sr_tokens_full[s0:s1]
                    sr_offset = s0
                    if sr_boundaries is not None:
                        sr_boundaries = {
                            (b - s0) for b in set(sr_boundaries) if s0 <= b <= s1
                        }

            return {
                "ur_tokens": ur_tokens,
                "sr_tokens": sr_tokens,
                "sr_boundaries": sr_boundaries,
                "ur_boundary_starts": ur_boundary_starts,
                "ur_offset": ur_offset,
                "sr_offset": sr_offset,
                "allowed_span_abs": allowed_span_abs,
                "allowed_span_slice": allowed_span_slice,
                "local_span_abs": local_span_abs,
            }

        for ur, sr, wid in self.all_examples:
            if ur == sr:
                continue
            if (
                self.allowed_morpheme is not None
                and wid not in self._morpheme_to_wids.get(self.allowed_morpheme, ())
            ):
                continue
            ur_tokens_full = list(ur.phonemes)
            sr_tokens = list(sr.phonemes)
            sr_boundaries = []
            if wid in self.segmentation:
                sr_boundaries = self.segmentation[wid][1:]
            ur_boundary_starts = set(
                (self._effective_boundary_starts_by_wid or {}).get(wid, set())
            )

            win = _slice_local_window_if_needed(
                wid,
                ur_tokens_full,
                sr_tokens,
                sr_boundaries,
                ur_boundary_starts,
            )
            ur_tokens = win["ur_tokens"]
            sr_tokens = win["sr_tokens"]
            sr_boundaries = win["sr_boundaries"]
            ur_boundary_starts = win["ur_boundary_starts"]
            ur_off = int(win.get("ur_offset", 0) or 0)
            local_abs = win.get("local_span_abs")

            # Compute an edit alignment over the full UR/SR token sequences,
            # and extract required insertions / deletions / transformations
            all_ops = _align_ops(ur_tokens, sr_tokens, bank=self.bank)

            # If there are multiple optimal alignments, consider all of them
            for ops in all_ops:
                for op, u_tok, s_tok, i_idx, j_idx in ops:
                    if self.allowed_morpheme is not None and local_abs is not None:
                        u0_abs, u1_abs = local_abs
                        i_abs = ur_off + int(i_idx)
                        if op in ("S", "D", "M"):
                            if not (u0_abs <= i_abs < u1_abs):
                                continue
                        elif op == "I":
                            if not (u0_abs <= i_abs <= u1_abs):
                                continue
                        elif op == "T":
                            if not (
                                (u0_abs <= i_abs < u1_abs)
                                or (u0_abs <= (i_abs + 1) < u1_abs)
                            ):
                                continue
                        else:
                            pass

                    if op == "M":
                        # match: no required change
                        continue
                    elif op == "T":
                        # metathesis (adjacent swap)
                        u_pair = u_tok
                        s_pair = s_tok
                        if (
                            isinstance(u_pair, tuple)
                            and isinstance(s_pair, tuple)
                            and len(u_pair) == 2
                            and len(s_pair) == 2
                            and u_pair[0] in self.bank.phonemes
                            and u_pair[1] in self.bank.phonemes
                            and s_pair[0] in self.bank.phonemes
                            and s_pair[1] in self.bank.phonemes
                        ):
                            i_abs = ur_off + int(i_idx)
                            L = ur_tokens_full[i_abs - 1] if i_abs > 0 else ""
                            R = (
                                ur_tokens_full[i_abs + 2]
                                if (i_abs + 2) < len(ur_tokens_full)
                                else ""
                            )
                            obs = Observation(id, u_pair, s_pair, L, R)
                            # Morpheme boundary metadata for guard generation.
                            #
                            # IMPORTANT: metathesis can move segments relative to SR morpheme
                            # boundaries, so SR boundary indices may not reliably reflect the
                            # underlying (morpheme) boundary location at the time the rule
                            # applies. When available, prefer UR boundary starts (token
                            # indices in the effective UR) to annotate possible "+" guards.
                            #
                            # Fallback to SR boundaries when UR boundaries are unavailable.
                            used_ur_boundaries = False
                            if ur_boundary_starts is not None and len(ur_boundary_starts) > 0:
                                used_ur_boundaries = True
                                # boundary immediately before the swapped pair
                                if i_idx in ur_boundary_starts:
                                    obs.morpheme_boundary_data["L"]["boundary_only"] = True
                                    if L:
                                        obs.morpheme_boundary_data["L"]["spec_plus_boundary"].add(
                                            L
                                        )
                                # boundary between the swapped phonemes
                                if (i_idx + 1) in ur_boundary_starts:
                                    obs.morpheme_boundary_data["L"]["boundary_only"] = True
                                    obs.morpheme_boundary_data["R"]["boundary_only"] = True
                                    if L:
                                        obs.morpheme_boundary_data["L"]["spec_plus_boundary"].add(
                                            L
                                        )
                                    if R:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_plus_spec"
                                        ].add(R)
                                # boundary immediately after the swapped pair
                                if (i_idx + 2) in ur_boundary_starts:
                                    obs.morpheme_boundary_data["R"]["boundary_only"] = True
                                    if R:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_plus_spec"
                                        ].add(R)

                            if (not used_ur_boundaries) and sr_boundaries:
                                # boundary immediately before the swapped pair
                                if j_idx in sr_boundaries:
                                    obs.morpheme_boundary_data["L"]["boundary_only"] = True
                                    if L:
                                        obs.morpheme_boundary_data["L"]["spec_plus_boundary"].add(
                                            L
                                        )
                                # boundary between the swapped phonemes
                                if (j_idx + 1) in sr_boundaries:
                                    obs.morpheme_boundary_data["L"]["boundary_only"] = True
                                    obs.morpheme_boundary_data["R"]["boundary_only"] = True
                                    if L:
                                        obs.morpheme_boundary_data["L"]["spec_plus_boundary"].add(
                                            L
                                        )
                                    if R:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_plus_spec"
                                        ].add(R)
                                # boundary immediately after the swapped pair
                                if (j_idx + 2) in sr_boundaries:
                                    obs.morpheme_boundary_data["R"]["boundary_only"] = True
                                    if R:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_plus_spec"
                                        ].add(R)
                    elif op == "I":
                        # insertion
                        if s_tok in self.bank.phonemes:
                            i_abs = ur_off + int(i_idx)
                            L = ur_tokens_full[i_abs - 1] if i_abs > 0 else ""
                            R = (
                                ur_tokens_full[i_abs]
                                if i_abs < len(ur_tokens_full)
                                else ""
                            )
                            obs = Observation(id, "", s_tok, L, R)
                            if sr_boundaries:
                                if j_idx in sr_boundaries:
                                    obs.morpheme_boundary_data["L"][
                                        "boundary_only"
                                    ] = True
                                    if L:
                                        obs.morpheme_boundary_data["L"][
                                            "spec_plus_boundary"
                                        ].add(L)
                                if (j_idx + 1) in sr_boundaries:
                                    obs.morpheme_boundary_data["R"][
                                        "boundary_only"
                                    ] = True
                                    if R:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_plus_spec"
                                        ].add(R)
                    elif op == "D":
                        # deletion
                        if u_tok in self.bank.phonemes:
                            i_abs = ur_off + int(i_idx)
                            L = ur_tokens_full[i_abs - 1] if i_abs > 0 else ""
                            R = (
                                ur_tokens_full[i_abs + 1]
                                if (i_abs + 1) < len(ur_tokens_full)
                                else ""
                            )
                            obs = Observation(id, u_tok, "", L, R)
                            if sr_boundaries:
                                if ur_boundary_starts is not None:
                                    if i_idx in ur_boundary_starts:
                                        obs.morpheme_boundary_data["L"][
                                            "boundary_only"
                                        ] = True
                                        if L:
                                            obs.morpheme_boundary_data["L"][
                                                "spec_plus_boundary"
                                            ].add(L)
                                    if (i_idx + 1) in ur_boundary_starts:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_only"
                                        ] = True
                                        if R:
                                            obs.morpheme_boundary_data["R"][
                                                "boundary_plus_spec"
                                            ].add(R)
                                else:
                                    if (j_idx - 1) in sr_boundaries:
                                        obs.morpheme_boundary_data["L"][
                                            "boundary_only"
                                        ] = True
                                        if L:
                                            obs.morpheme_boundary_data["L"][
                                                "spec_plus_boundary"
                                            ].add(L)

                                    if j_idx in sr_boundaries:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_only"
                                        ] = True
                                        if R:
                                            obs.morpheme_boundary_data["R"][
                                                "boundary_plus_spec"
                                            ].add(R)
                    elif op == "S":
                        # substitution
                        if u_tok in self.bank.phonemes and s_tok in self.bank.phonemes:
                            i_abs = ur_off + int(i_idx)
                            L = ur_tokens_full[i_abs - 1] if i_abs > 0 else ""
                            R = (
                                ur_tokens_full[i_abs + 1]
                                if (i_abs + 1) < len(ur_tokens_full)
                                else ""
                            )
                            obs = Observation(id, u_tok, s_tok, L, R)
                            if sr_boundaries:
                                if j_idx in sr_boundaries:
                                    obs.morpheme_boundary_data["L"][
                                        "boundary_only"
                                    ] = True
                                    if L:
                                        obs.morpheme_boundary_data["L"][
                                            "spec_plus_boundary"
                                        ].add(L)
                                if (j_idx + 1) in sr_boundaries:
                                    obs.morpheme_boundary_data["R"][
                                        "boundary_only"
                                    ] = True
                                    if R:
                                        obs.morpheme_boundary_data["R"][
                                            "boundary_plus_spec"
                                        ].add(R)
                    id += 1
                    observations.append(obs)
        return observations

    def _expand(self, candidates, cost, freq_cap=None, use_features=True):
        """
        Expand candidates by extracting features from phonemes.
        Keep only freq_cap most frequent candidates.
        """
        expanded = defaultdict(int)
        key_to_spec = {}

        def _spec_key_local(spec):
            if spec is None:
                return ("N",)
            if isinstance(spec, ConstantPhoneme):
                return ("C", spec.p)
            if isinstance(spec, FeatureMatrix):
                return ("F", tuple(spec.featuresAndPolarities))
            return ("S", str(spec))

        for p, count in candidates.items():
            # OffsetSpecification
            if p in ["-1", "1"]:
                if cost >= OffsetSpecification(int(p)).cost():
                    expanded[p] += count
                    key_to_spec[p] = OffsetSpecification(int(p))
                continue
            # MetathesisFocus
            if isinstance(p, tuple) and len(p) == 2:
                # Concrete metathesis focus (original behavior): swap this *specific*
                # adjacent pair (p0 p1) -> (p1 p0).
                mf_concrete = MetathesisFocus(ConstantPhoneme(p[0]), ConstantPhoneme(p[1]))
                if cost >= mf_concrete.cost():
                    k = ("MF", _spec_key_local(mf_concrete.s1), _spec_key_local(mf_concrete.s2))
                    expanded[k] += count
                    key_to_spec[k] = mf_concrete

                # Index-based metathesis focus (new behavior): swap the 1st and 2nd
                # phoneme *after the left guard* (value-agnostic). We represent this as
                # MetathesisFocus(None, None), which matches any adjacent pair; the
                # left/right guards determine where it applies.
                mf_idx = MetathesisFocus(None, None)
                if cost >= mf_idx.cost():
                    k = ("MF_IDX", 1, 2)
                    expanded[k] += count
                    key_to_spec[k] = mf_idx
                continue
            # ConstantPhoneme
            if cost >= ConstantPhoneme(p).cost():
                expanded[p] += count
                key_to_spec[p] = ConstantPhoneme(p)
            # FeatureMatrix
            if not use_features:
                continue
            for f, v in zip(self.bank.features, self.bank.featureVectorMap.get(p, [])):
                if cost >= FeatureMatrix([(v, f)]).cost():
                    expanded[f"({v}, {f})"] += count
                    key_to_spec[f"({v}, {f})"] = FeatureMatrix([(v, f)])
        expanded = sorted(expanded.items(), key=lambda x: x[1], reverse=True)[:freq_cap]
        return [key_to_spec[k] for k, _ in expanded]

    def _generate_focus_candidates(self, cost, rule_type, freq_cap=None):
        """
        Generate valid focus candidates.

        Args:
            cost: float
            rule_type: "insertion" | "deletion" | "substitution"
            freq_cap: int | None
        """
        candidates = defaultdict(int)
        for obs in self.observations:
            X, Y, L, R = obs.context
            if (
                rule_type == "metathesis"
                and isinstance(X, tuple)
                and isinstance(Y, tuple)
                and len(X) == 2
                and len(Y) == 2
            ):
                if X[0] != X[1] and Y == (X[1], X[0]):
                    candidates[X] += 1
            if rule_type == "insertion" and X == "" and Y != "":
                if cost >= EmptySpecification().cost() and (
                    freq_cap is None or freq_cap > 0
                ):
                    return [EmptySpecification()]
                else:
                    return []
            elif rule_type == "deletion" and X != "" and Y == "":
                candidates[X] += 1
            elif rule_type == "substitution" and X != "" and Y != "":
                candidates[X] += 1
        return self._expand(candidates, cost, freq_cap)

    def _compatible(self, x, y):
        """
        Check if x is accurately described by y.

        Args:
            x: str
            y: Specification
        """
        # wildcard (used by index-based metathesis focus)
        if y is None:
            return True
        # empty alias
        if x == "" and str(y) == "Ø":
            return True
        # phoneme alias
        if x != "" and str(y) == x:
            return True
        # feature contained in x
        if isinstance(y, FeatureMatrix):
            for v, f in y.featuresAndPolarities:
                if (f, v) not in zip(
                    self.bank.features, self.bank.featureVectorMap.get(x, [])
                ):
                    return False
            return True
        # offsets are checked separately
        if isinstance(y, OffsetSpecification):
            return True
        # metathesis focus
        if isinstance(y, MetathesisFocus):
            if not isinstance(x, tuple) or len(x) != 2:
                return False
            return self._compatible(x[0], y.s1) and self._compatible(x[1], y.s2)
        # metathesis change (checked separately)
        if isinstance(y, MetathesisSpecification):
            return True
        return False

    def _generate_change_candidates(self, cost, rule_type, focus, freq_cap=None):
        """
        Generate valid change candidates.

        Args:
            cost: float
            rule_type: "insertion" | "deletion" | "substitution"
            focus: Specification
            freq_cap: int | None
        """
        remaining_cost = cost - focus.cost()
        candidates = defaultdict(int)
        for obs in self.observations:
            X, Y, L, R = obs.context
            if not self._compatible(X, focus):
                continue
            if (
                rule_type == "metathesis"
                and isinstance(X, tuple)
                and isinstance(Y, tuple)
                and len(X) == 2
                and len(Y) == 2
            ):
                if remaining_cost >= MetathesisSpecification().cost():
                    return [MetathesisSpecification()]
                else:
                    return []
            if rule_type == "deletion" and X != "" and Y == "":
                if remaining_cost >= EmptySpecification().cost() and (
                    freq_cap is None or freq_cap > 0
                ):
                    return [EmptySpecification()]
                else:
                    return []
            elif rule_type == "insertion" and X == "" and Y != "":
                candidates[Y] += 1
            elif rule_type == "substitution" and X != "" and Y != "":
                candidates[Y] += 1
            # copy rules
            if Y == L:
                candidates["-1"] += 1
            if Y == R:
                candidates["1"] += 1
        return self._expand(
            candidates,
            remaining_cost,
            freq_cap,
            use_features=(rule_type != "insertion"),
        )

    def _generate_guard_candidates(self, cost, focus, change, freq_cap=None):
        """
        Generate valid guard candidates.

        Args:
            cost: float
            focus: Specification
            change: Specification
            freq_cap: int | None
        """
        remaining_cost = cost - focus.cost() - change.cost()

        # Generate all left-right pairs jointly supported by >=1 observation;
        # maintain global frequency stats over such pairs
        pairs = set()

        left_guards = defaultdict(int)
        right_guards = defaultdict(int)

        key_to_spec_left = {}
        key_to_spec_right = {}

        for obs in self.observations:
            X, Y, L, R = obs.context
            if not self._compatible(X, focus):
                continue
            if not self._compatible(Y, change):
                continue
            if change == OffsetSpecification(-1) and Y != L:
                continue
            if change == OffsetSpecification(1) and Y != R:
                continue
            if (
                isinstance(focus, MetathesisFocus)
                and isinstance(change, MetathesisSpecification)
                and not (
                    isinstance(X, tuple)
                    and isinstance(Y, tuple)
                    and len(X) == 2
                    and len(Y) == 2
                    and Y == (X[1], X[0])
                )
            ):
                continue

            # Empty guards
            left_keys = [""]
            key_to_spec_left[""] = Guard("L", False, False, False, [])

            right_keys = [""]
            key_to_spec_right[""] = Guard("R", False, False, False, [])

            # Base guards (one spec, no boundary)
            if L:
                left_keys.append(L)
                key_to_spec_left[L] = Guard(
                    "L", False, False, False, [ConstantPhoneme(L)]
                )
                for f, v in zip(
                    self.bank.features, self.bank.featureVectorMap.get(L, [])
                ):
                    left_keys.append(f"({v}, {f})")
                    key_to_spec_left[f"({v}, {f})"] = Guard(
                        "L", False, False, False, [FeatureMatrix([(v, f)])]
                    )
            if R:
                right_keys.append(R)
                key_to_spec_right[R] = Guard(
                    "R", False, False, False, [ConstantPhoneme(R)]
                )
                for f, v in zip(
                    self.bank.features, self.bank.featureVectorMap.get(R, [])
                ):
                    right_keys.append(f"({v}, {f})")
                    key_to_spec_right[f"({v}, {f})"] = Guard(
                        "R", False, False, False, [FeatureMatrix([(v, f)])]
                    )

            sup = obs.morpheme_boundary_data

            # Boundary-only guards
            if sup["L"]["boundary_only"]:
                left_keys.append("+")
                key_to_spec_left["+"] = Guard(
                    "L", False, False, False, [BoundarySpecification()]
                )

            if sup["R"]["boundary_only"]:
                right_keys.append("+")
                key_to_spec_right["+"] = Guard(
                    "R", False, False, False, [BoundarySpecification()]
                )

            # Boundary + spec combos (both orders)
            for p in sup["L"]["spec_plus_boundary"]:
                left_keys.append(f"{p} +")
                key_to_spec_left[f"{p} +"] = Guard(
                    "L",
                    False,
                    False,
                    False,
                    [BoundarySpecification(), ConstantPhoneme(p)],
                )
                for f, v in zip(
                    self.bank.features, self.bank.featureVectorMap.get(p, [])
                ):
                    left_keys.append(f"({v}, {f}) +")
                    key_to_spec_left[f"({v}, {f}) +"] = Guard(
                        "L",
                        False,
                        False,
                        False,
                        [BoundarySpecification(), FeatureMatrix([(v, f)])],
                    )

            for p in sup["R"]["boundary_plus_spec"]:
                right_keys.append(f"{p} +")
                key_to_spec_right[f"{p} +"] = Guard(
                    "R",
                    False,
                    False,
                    False,
                    [BoundarySpecification(), ConstantPhoneme(p)],
                )
                for f, v in zip(
                    self.bank.features, self.bank.featureVectorMap.get(p, [])
                ):
                    right_keys.append(f"({v}, {f}) +")
                    key_to_spec_right[f"({v}, {f}) +"] = Guard(
                        "R",
                        False,
                        False,
                        False,
                        [BoundarySpecification(), FeatureMatrix([(v, f)])],
                    )

            for lk in left_keys:
                left_guards[lk] += 1
            for rk in right_keys:
                right_guards[rk] += 1

            # Form pairs by cross-product
            for lk in left_keys:
                for rk in right_keys:
                    # For local rules, only allow rules that refer to morpheme boundaries
                    if self.allowed_morpheme is not None:
                        if not any(
                            isinstance(s, BoundarySpecification)
                            for s in key_to_spec_left[lk].specifications
                        ) and not any(
                            isinstance(s, BoundarySpecification)
                            for s in key_to_spec_right[rk].specifications
                        ):
                            continue
                    if (
                        key_to_spec_left[lk].cost() + key_to_spec_right[rk].cost()
                        <= remaining_cost
                    ):
                        pairs.add((lk, rk))

        pairs = list(pairs)
        pairs = sorted(
            pairs,
            key=lambda x: left_guards[x[0]] * right_guards[x[1]],
            reverse=True,
        )[:freq_cap]
        return [(key_to_spec_left[lk], key_to_spec_right[rk]) for lk, rk in pairs]

        # OLD
        # endOfString guards (word boundary #).
        # endOfString itself has no feature cost; guards are gated by their specifications' feature cost.
        # Try with single phoneme
        # if cost >= 1.0:
        #     for p in list(contexts)[:3] if contexts else list(example_phonemes)[:3]:
        #         if p in self.bank.phonemes:
        #             g = Guard("L" if is_left else "R", True, False, False, [ConstantPhoneme(p)])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        # Try with single feature
        # if cost >= 1.0:
        #     for f_name in list(context_features)[:3] if context_features else []:
        #         if f_name in self.bank.features:
        #             g = Guard("L" if is_left else "R", True, False, False,
        #                      [FeatureMatrix([(True, f_name)])])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        # [-f] at word boundary only if some phoneme lacks f
        # if f_name in negative_eligible_features:
        #     g = Guard(
        #         "L" if is_left else "R",
        #         True,
        #         False,
        #         False,
        #         [FeatureMatrix([(False, f_name)])],
        #     )
        #     if g.cost() <= cost:
        #         candidates.append(g)

        # optionalEnding guards (optional end of string).
        # optionalEnding itself has no feature cost; guards are gated by their specifications' feature cost.
        # Try with single phoneme
        # if cost >= 1.0:
        #     for p in list(contexts)[:2] if contexts else list(example_phonemes)[:2]:
        #         if p in self.bank.phonemes:
        #             g = Guard("L" if is_left else "R", False, True, False, [ConstantPhoneme(p)])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        # Try with single feature
        # if cost >= 1.0:
        #     for f_name in list(context_features)[:2] if context_features else []:
        #         if f_name in self.bank.features:
        #             g = Guard("L" if is_left else "R", False, True, False,
        #                      [FeatureMatrix([(True, f_name)])])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        # [-f] with optional ending only if some phoneme lacks f
        # if f_name in negative_eligible_features:
        #     g = Guard(
        #         "L" if is_left else "R",
        #         False,
        #         True,
        #         False,
        #         [FeatureMatrix([(False, f_name)])],
        #     )
        #     if g.cost() <= cost:
        #         candidates.append(g)

        # Guards with 2 specifications (spec spec2).
        # Under the new metric, 2-feature guards have feature cost 2.
        # Try 2 phonemes (typically more expensive, so require higher budget)
        # if cost >= 4.0:
        #     context_list = list(contexts)[:3] if contexts else list(example_phonemes)[:3]
        #     for i, p1 in enumerate(context_list):
        #         if p1 not in self.bank.phonemes:
        #             continue
        #         for p2 in context_list[i+1:]:
        #             if p2 not in self.bank.phonemes:
        #                 continue
        #             g = Guard("L" if is_left else "R", False, False, False,
        #                      [ConstantPhoneme(p1), ConstantPhoneme(p2)])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        #             if len([c for c in candidates if len(c.specifications) == 2]) >= 3:  # Limit
        #                 break
        #         if len([c for c in candidates if len(c.specifications) == 2]) >= 3:
        #             break
        # Try 2 features
        # if cost >= 2.0:
        #     feature_list = list(context_features)[:3] if context_features else []
        #     for i, f1 in enumerate(feature_list):
        #         if f1 not in self.bank.features:
        #             continue
        #         for f2 in feature_list[i+1:]:
        #             if f2 not in self.bank.features:
        #                 continue
        #             g = Guard("L" if is_left else "R", False, False, False,
        #                      [FeatureMatrix([(True, f1)]), FeatureMatrix([(True, f2)])])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        #             if len([c for c in candidates if len(c.specifications) == 2 and
        #                    all(isinstance(s, FeatureMatrix) for s in c.specifications)]) >= 2:
        #                 break
        #         if len([c for c in candidates if len(c.specifications) == 2 and
        #                all(isinstance(s, FeatureMatrix) for s in c.specifications)]) >= 2:
        #             break

        # Starred guards (spec*spec2 - zero or more occurrences).
        # Requires 2 specifications; star has no feature cost in this metric.
        # For 2-phoneme guards, keep a higher budget threshold.
        # if cost >= 5.0:
        #     context_list = list(contexts)[:2] if contexts else list(example_phonemes)[:2]
        #     for i, p1 in enumerate(context_list):
        #         if p1 not in self.bank.phonemes:
        #             continue
        #         for p2 in context_list[i+1:]:
        #             if p2 not in self.bank.phonemes:
        #                 continue
        #             # Starred: spec*spec2 (p1* p2)
        #             g = Guard("L" if is_left else "R", False, False, True,
        #                      [ConstantPhoneme(p1), ConstantPhoneme(p2)])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        #             if len([c for c in candidates if c.starred]) >= 2:  # Limit
        #                 break
        #         if len([c for c in candidates if c.starred]) >= 2:
        #             break
        # 2 features + starred: requires budget for at least 2 feature units.
        # if cost >= 2.0:
        #     feature_list = list(context_features)[:2] if context_features else []
        #     for i, f1 in enumerate(feature_list):
        #         if f1 not in self.bank.features:
        #             continue
        #         for f2 in feature_list[i+1:]:
        #             if f2 not in self.bank.features:
        #                 continue
        #             g = Guard("L" if is_left else "R", False, False, True,
        #                      [FeatureMatrix([(True, f1)]), FeatureMatrix([(True, f2)])])
        #             if g.cost() <= cost:
        #                 candidates.append(g)
        #             if len([c for c in candidates if c.starred and
        #                    all(isinstance(s, FeatureMatrix) for s in c.specifications)]) >= 1:
        #                 break
        #         if len([c for c in candidates if c.starred and
        #                all(isinstance(s, FeatureMatrix) for s in c.specifications)]) >= 1:
        #             break

        # return candidates

    def topK(self, k=1, max_cost=10.0):
        """Find up to k best rules using constraint-based solving."""
        prof = getattr(self, "_prof", None) or RuntimeProfiler(enabled=False)
        if prof.enabled:
            prof.reset()

        top_n = int(os.getenv("BPL_RULE_SEARCH_PROFILE_TOP", "30"))
        min_ms = float(os.getenv("BPL_RULE_SEARCH_PROFILE_MIN_MS", "50"))
        report_every_s = float(os.getenv("BPL_RULE_SEARCH_PROFILE_EVERY_S", "60"))
        last_report_t = time.perf_counter()

        def _maybe_report(tag="partial"):
            nonlocal last_report_t
            if not prof.enabled or report_every_s <= 0:
                return
            now = time.perf_counter()
            if (now - last_report_t) >= report_every_s:
                prof.report(
                    header=f"=== RuleSolver.topK runtime profile ({tag}) ===",
                    top_n=top_n,
                    min_total_ms=min_ms,
                )
                last_report_t = now

        best_rules = []
        ed = EditDistanceCalculator(self.bank)

        guard_cache = {}
        guard_cache_max = int(os.getenv("BPL_GUARD_CACHE_MAX", "200000"))

        baseline_tfed = 0.0
        baseline_dists = []
        sr_norm = []

        ur_phoneme_sets = []
        for ur0_eff, sr0, _wid0 in self.all_examples:
            ur0_n = ed.normalize_input(ur0_eff)
            sr0_n = ed.normalize_input(sr0)
            sr_norm.append(sr0_n)

            d0 = ed.edit_distance(ur0_n, sr0_n)
            baseline_dists.append(d0)
            baseline_tfed += d0

            if hasattr(ur0_eff, "phonemes"):
                ur_phoneme_sets.append(frozenset(ur0_eff.phonemes))
            else:
                ur_phoneme_sets.append(frozenset(tokenize(ur0_n)))

        focus_to_candidate_indices = {}
        spec_to_phonemes = {}
        bank_phonemes = tuple(self.bank.phonemes)

        def _spec_key(spec):
            if spec is None:
                return ("N",)
            if isinstance(spec, ConstantPhoneme):
                return ("C", spec.p)
            if isinstance(spec, FeatureMatrix):
                return ("F", tuple(spec.featuresAndPolarities))
            if isinstance(spec, EmptySpecification):
                return ("E",)
            if isinstance(spec, OffsetSpecification):
                return ("O", int(spec.offset))
            if isinstance(spec, BoundarySpecification):
                return ("B",)
            if isinstance(spec, PlaceSpecification):
                return ("P",)
            if isinstance(spec, MetathesisFocus):
                return ("MF", _spec_key(spec.s1), _spec_key(spec.s2))
            if isinstance(spec, MetathesisSpecification):
                return ("MS",)
            return ("S", str(spec))

        def _phonemes_matching_spec(spec):
            """Return a set of phoneme tokens that can match `spec`."""
            if spec is None:
                return None
            # ConstantPhoneme matches itself
            if isinstance(spec, ConstantPhoneme):
                return frozenset([spec.p])
            # FeatureMatrix matches the phonemes that contain it as a subset
            if isinstance(spec, FeatureMatrix):
                k = _spec_key(spec)
                cached = spec_to_phonemes.get(k)
                if cached is not None:
                    return cached
                matches = [p for p in bank_phonemes if matches_specification(p, spec)]
                res = frozenset(matches)
                spec_to_phonemes[k] = res
                return res
            # For other specs (offset/boundary/empty),
            # any phoneme can match or we cannot safely prune
            return None

        def _candidate_indices_for_focus(focus_spec):
            """
            Return indices into `self.all_examples` that could possibly be affected by a
            rule with this focus.
            """
            if isinstance(focus_spec, EmptySpecification):
                return tuple(range(len(self.all_examples)))

            if isinstance(focus_spec, MetathesisFocus):
                s1 = _phonemes_matching_spec(focus_spec.s1)
                s2 = _phonemes_matching_spec(focus_spec.s2)
                if s1 is None or s2 is None:
                    return tuple(range(len(self.all_examples)))
                idxs = []
                for i, (ur_eff_i, _sr_i, _wid_i) in enumerate(self.all_examples):
                    phs = getattr(ur_eff_i, "phonemes", None)
                    if not phs or len(phs) < 2:
                        continue
                    ok = False
                    for j in range(len(phs) - 1):
                        if phs[j] in s1 and phs[j + 1] in s2:
                            ok = True
                            break
                    if ok:
                        idxs.append(i)
                return tuple(idxs)

            phs_ok = _phonemes_matching_spec(focus_spec)
            if phs_ok is None:
                return tuple(range(len(self.all_examples)))

            idxs = []
            for i, phset in enumerate(ur_phoneme_sets):
                if phset and (phset & phs_ok):
                    idxs.append(i)
            return tuple(idxs)

        with prof.span("topK.total"):
            for rule_type in ["insertion", "deletion", "substitution", "metathesis"]:
                _maybe_report(tag=f"rule_type={rule_type}")
                with prof.span("candidates.focus"):
                    focus_candidates = self._generate_focus_candidates(
                        max_cost, rule_type
                    )

                for focus in focus_candidates:
                    _maybe_report(tag=f"rule_type={rule_type}")
                    with prof.span("candidates.change"):
                        change_candidates = self._generate_change_candidates(
                            max_cost, rule_type, focus
                        )

                    for change in change_candidates:
                        _maybe_report(tag=f"rule_type={rule_type}")
                        with prof.span("candidates.guards"):
                            guards = self._generate_guard_candidates(
                                max_cost, focus, change
                            )

                        for left_guard, right_guard in guards:
                            # For local rules, we explore two modes:
                            #   - "allowed_morpheme": apply inside the allowed morpheme;
                            #   - "neighbor": apply in its neighbor(s).
                            # Special case: if '+' appears in both guards,
                            # only "allowed_morpheme" is considered.
                            if self.allowed_morpheme is not None:
                                left_has_boundary = any(
                                    isinstance(s, BoundarySpecification)
                                    for s in left_guard.specifications
                                )
                                right_has_boundary = any(
                                    isinstance(s, BoundarySpecification)
                                    for s in right_guard.specifications
                                )
                                if left_has_boundary and right_has_boundary:
                                    locality_modes = ["allowed_morpheme"]
                                else:
                                    locality_modes = ["allowed_morpheme", "neighbor"]
                            else:
                                locality_modes = [None]

                            for locality_mode in locality_modes:
                                _maybe_report(tag=f"rule_type={rule_type}")
                                rule = Rule(focus, change, left_guard, right_guard)
                                setattr(rule, "locality_mode", locality_mode)
                                if rule.cost() > max_cost:
                                    continue

                                rule_tfed = baseline_tfed
                                had_error = False

                                with prof.span("rule_eval.all_examples"):
                                    focus_key = _spec_key(focus)
                                    candidate_indices = focus_to_candidate_indices.get(
                                        focus_key
                                    )
                                    if candidate_indices is None:
                                        candidate_indices = (
                                            _candidate_indices_for_focus(focus)
                                        )
                                        focus_to_candidate_indices[focus_key] = (
                                            candidate_indices
                                        )

                                    for idx in candidate_indices:
                                        (ur_eff, sr, wid) = self.all_examples[idx]
                                        try:
                                            boundary_positions_eff = self._effective_boundary_positions_by_wid.get(
                                                wid
                                            )
                                            spans_eff = (
                                                self._effective_spans_by_wid.get(wid)
                                            )
                                            t_apply0 = (
                                                time.perf_counter()
                                                if prof.enabled
                                                else None
                                            )
                                            result = apply_rule(
                                                rule,
                                                ur_eff,
                                                bank=self.bank,
                                                allowed_morpheme=self.allowed_morpheme,
                                                morphemes=self.morphemes,
                                                spans_override=spans_eff,
                                                boundary_positions_override=boundary_positions_eff,
                                                order=self.order,
                                                word_to_slot_values=self.word_to_slot_values,
                                                wid=wid,
                                                guard_cache=guard_cache,
                                                guard_cache_max=guard_cache_max,
                                            )
                                            if prof.enabled:
                                                prof.add_time(
                                                    "rule_eval.apply_rule",
                                                    time.perf_counter() - t_apply0,
                                                )
                                                prof.inc("rule_eval.apply_rule_calls")

                                            if result is ur_eff:
                                                continue

                                            t_ed0 = (
                                                time.perf_counter()
                                                if prof.enabled
                                                else None
                                            )
                                            res_norm = ed.normalize_input(result)
                                            new_dist = ed.edit_distance(
                                                res_norm,
                                                sr_norm[idx],
                                            )

                                            rule_tfed += new_dist - baseline_dists[idx]
                                            if prof.enabled:
                                                prof.add_time(
                                                    "rule_eval.edit_distance",
                                                    time.perf_counter() - t_ed0,
                                                )
                                                prof.inc(
                                                    "rule_eval.edit_distance_calls"
                                                )

                                        except Exception:
                                            traceback.print_exc()
                                            had_error = True
                                            break

                                if had_error:
                                    continue

                                if rule_tfed < baseline_tfed:
                                    best_rules.append((rule, rule_tfed, rule.cost()))
                                    best_rules.sort(key=lambda r: (r[1], r[2]))
                                    if len(best_rules) > k:
                                        best_rules = best_rules[:k]

        if prof.enabled:
            prof.report(
                header="=== RuleSolver.topK runtime profile (wall clock) ===",
                top_n=top_n,
                min_total_ms=min_ms,
            )

        return [r for (r, _, _) in best_rules]
