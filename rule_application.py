from morph import Morph
from rule import (
    ConstantPhoneme,
    FeatureMatrix,
    EmptySpecification,
    OffsetSpecification,
    PlaceSpecification,
    BoundarySpecification,
    Guard,
    MetathesisFocus,
    MetathesisSpecification,
)
from features import FeatureBank, featureMap, tokenize


def matches_specification(phoneme, spec):
    """Check if a phoneme matches a specification."""
    if spec is None:
        return True
    if isinstance(spec, ConstantPhoneme):
        return phoneme == spec.p
    elif isinstance(spec, FeatureMatrix):
        features = featureMap.get(phoneme, [])
        for polarity, feature in spec.featuresAndPolarities:
            if polarity:
                if feature not in features:
                    return False
            else:
                if feature in features:
                    return False
        return True
    elif isinstance(spec, BoundarySpecification):
        # Boundary specifications are handled explicitly in guard application
        # using morpheme boundary metadata. They should not be matched against
        # individual phonemes.
        return False
    elif isinstance(spec, OffsetSpecification):
        return True  # Offset matches any phoneme
    elif isinstance(spec, PlaceSpecification):
        return True  # Place matches any phoneme
    else:
        return False


def apply_specification(phoneme, spec, bank):
    """Apply a specification to a phoneme, returning the resulting phoneme."""
    if spec is None:
        return phoneme
    if isinstance(spec, ConstantPhoneme):
        return spec.p
    elif isinstance(spec, FeatureMatrix):
        # Apply feature changes
        features = list(featureMap.get(phoneme, []))
        for polarity, feature in spec.featuresAndPolarities:
            if polarity:
                if feature not in features:
                    features.append(feature)
                # Remove mutually exclusive features
                for exclusive_class in FeatureBank.mutuallyExclusiveClasses:
                    if feature in exclusive_class:
                        features = [f for f in features if f not in exclusive_class or f == feature]
                        break
            else:
                features = [f for f in features if f != feature]
        # Find phoneme with matching features
        target_features = frozenset(features)
        return bank.matrix2phoneme.get(target_features, phoneme)
    elif isinstance(spec, OffsetSpecification):
        # This shouldn't be called directly for offset
        raise ValueError("Cannot apply offset specification directly")
    elif isinstance(spec, PlaceSpecification):
        # Place assimilation - simplified version
        return phoneme
    else:
        return phoneme


def _spec_cache_key(spec):
    """Stable, hashable cache key for a guard specification."""
    if spec is None:
        return ("N",)
    if isinstance(spec, ConstantPhoneme):
        return ("C", spec.p)
    if isinstance(spec, FeatureMatrix):
        return ("F", tuple(getattr(spec, "featuresAndPolarities", []) or []))
    if isinstance(spec, BoundarySpecification):
        return ("B",)
    if isinstance(spec, OffsetSpecification):
        return ("O", int(getattr(spec, "offset", 0)))
    if isinstance(spec, PlaceSpecification):
        return ("P",)
    if isinstance(spec, EmptySpecification):
        return ("E",)
    if isinstance(spec, MetathesisFocus):
        return ("MF", _spec_cache_key(spec.s1), _spec_cache_key(spec.s2))
    if isinstance(spec, MetathesisSpecification):
        return ("MS",)
    return ("S", str(spec))


def _guard_cache_key(phonemes, guard, right_side, boundary_positions):
    specs = tuple(_spec_cache_key(s) for s in (getattr(guard, "specifications", []) or []))
    return (
        tuple(phonemes),
        frozenset(boundary_positions or []),
        bool(right_side),
        getattr(guard, "side", None),
        bool(getattr(guard, "endOfString", False)),
        bool(getattr(guard, "optionalEnding", False)),
        bool(getattr(guard, "starred", False)),
        specs,
    )


def apply_guard_sophisticated(phonemes, guard, right_side, bank, boundary_positions=None, guard_cache=None, guard_cache_max=None):
    """
    Apply a guard to a sequence of phonemes.
    Returns a list of booleans indicating at each position whether the guard matches.
    
    For right_side=False (left guard): okay[j] means guard matches phonemes[0:j]
    For right_side=True (right guard): okay[j] means guard matches phonemes[j:]
    """
    cache_key = None
    if guard_cache is not None:
        try:
            cache_key = _guard_cache_key(phonemes, guard, right_side, boundary_positions)
            cached = guard_cache.get(cache_key)
            if cached is not None:
                return cached
        except Exception:
            cache_key = None

    def _finish(okay_list):
        res = tuple(okay_list)
        if guard_cache is not None and cache_key is not None:
            try:
                maxn = int(guard_cache_max) if guard_cache_max is not None else None
                if maxn is not None and maxn > 0 and len(guard_cache) >= maxn:
                    guard_cache.clear()
                guard_cache[cache_key] = res
            except Exception:
                pass
        return res

    l = len(phonemes)
    okay = [False] * (l + 1)

    boundary_positions = set(boundary_positions or [])
    has_boundary = any(isinstance(s, BoundarySpecification) for s in guard.specifications)

    # Special handling for guards that mention a morpheme boundary '+'.
    # We interpret '+' as marking a boundary between morphemes at specific
    # positions in the word, provided via boundary_positions (integers in [0, l]).
    #
    # We currently support up to one boundary specification per guard, optionally
    # combined with a single phoneme/feature specification, and without stars or
    # optional endings – exactly the patterns used for local morpheme-boundary
    # rules such as:
    #   - n > m | + b _
    #   - z > s | _ + [-voice]
    #   - t > d | [+voice] + _
    #   - b > m | _ + [+nasal]
    if has_boundary and boundary_positions:
        num_boundaries = sum(
            1 for s in guard.specifications if isinstance(s, BoundarySpecification)
        )
        non_boundary_specs = [
            s for s in guard.specifications if not isinstance(s, BoundarySpecification)
        ]

        # If the pattern is more complex than we currently support, fall back to
        # the generic implementation (which will effectively never match due to
        # BoundarySpecification.matches == False).
        if (
            num_boundaries == 1
            and len(non_boundary_specs) <= 1
            and not guard.starred
            and not guard.optionalEnding
            and not guard.endOfString
        ):
            other_spec = non_boundary_specs[0] if non_boundary_specs else None

            # Determine ordering when a second spec is present.
            # If there are 2 specs, boundary can be at index 0 or 1.
            # We interpret:
            #  - LEFT guard:
            #      [Boundary, S] => "S +"   (S before boundary, boundary adjacent to focus)
            #      [S, Boundary] => "+ S"   (boundary before S, boundary one segment away)
            #  - RIGHT guard:
            #      [Boundary, S] => "+ S"   (boundary adjacent to focus)
            #      [S, Boundary] => "S +"   (boundary one segment away)
            boundary_first = False
            if other_spec is not None and len(guard.specifications) == 2:
                boundary_first = isinstance(guard.specifications[0], BoundarySpecification)

            if not right_side:
                # LEFT GUARD at position j: prefix phonemes[0:j]
                if other_spec is None:
                    # Pattern: '+' (boundary immediately to the left of the focus)
                    for j in range(l + 1):
                        if j in boundary_positions:
                            okay[j] = True
                else:
                    if boundary_first:
                        # Pattern: "S +"  (S before boundary; boundary at j)
                        for j in range(l + 1):
                            if j in boundary_positions and j - 1 >= 0:
                                if matches_specification(phonemes[j - 1], other_spec):
                                    okay[j] = True
                    else:
                        # Pattern: "+ S"  (boundary before S; boundary at j-1)
                        for j in range(l + 1):
                            bpos = j - 1
                            if bpos in boundary_positions and j - 1 >= 0:
                                if matches_specification(phonemes[j - 1], other_spec):
                                    okay[j] = True
            else:
                # RIGHT GUARD at position j: suffix phonemes[j:]
                if other_spec is None:
                    # Pattern: '+' (boundary immediately to the right of the focus)
                    for j in range(l + 1):
                        if j in boundary_positions:
                            okay[j] = True
                else:
                    if boundary_first:
                        # Pattern: "+ S"  (boundary at j; S after boundary)
                        for j in range(l + 1):
                            if j in boundary_positions and j < l:
                                if matches_specification(phonemes[j], other_spec):
                                    okay[j] = True
                    else:
                        # Pattern: "S +"  (S before boundary; boundary at j+1)
                        for j in range(l + 1):
                            bpos = j + 1
                            if bpos in boundary_positions and j < l:
                                if matches_specification(phonemes[j], other_spec):
                                    okay[j] = True

            return _finish(okay)

    # Generic guard handling (no morpheme boundary, or we have no boundary
    # metadata available). This is a direct, feature-only port of the original
    # Sketch guard automaton.
    # Handle empty guard
    if guard.doesNothing():
        if guard.endOfString:
            okay[l if right_side else 0] = True
        else:
            # Empty guard matches everywhere
            for j in range(l + 1):
                okay[j] = True
        return _finish(okay)

    r1 = guard.specifications[1] if len(guard.specifications) > 1 else None
    r2 = guard.specifications[0] if len(guard.specifications) > 0 else None

    optional_ending_1 = guard.optionalEnding and r1 is not None
    optional_ending_2 = guard.optionalEnding and r2 is not None and r1 is None

    def index2sound(o):
        """Convert okay index to sound index."""
        if right_side:
            return l - 1 - (o - 1)
        else:
            return o - 1

    a1_old = False
    a2_old = False

    for j in range(l + 1):
        # Check r1 (second specification, if present)
        if r1 is None:
            a1 = (j == 0 or not guard.endOfString)
        else:
            if j > 0:
                sound_idx = index2sound(j)
                if 0 <= sound_idx < l:
                    a1 = (
                        matches_specification(phonemes[sound_idx], r1)
                        and (j == 1 or not guard.endOfString)
                    )
                else:
                    a1 = False
            else:
                a1 = False
            if optional_ending_1:
                a1 = a1 or (j == 0)

        # Check r2 (first specification)
        if r2 is None:
            a2 = a1
        elif guard.starred:
            # Starred: r2* matches zero or more occurrences
            if j > 0:
                sound_idx = index2sound(j)
                if 0 <= sound_idx < l:
                    a2 = a1 or (
                        matches_specification(phonemes[sound_idx], r2) and a2_old
                    )
                else:
                    a2 = a1
            else:
                a2 = a1
        else:
            # Not starred: r1 r2 must match
            if j > 0:
                sound_idx = index2sound(j)
                if 0 <= sound_idx < l:
                    a2 = a1_old and matches_specification(phonemes[sound_idx], r2)
                else:
                    a2 = False
            else:
                a2 = False
            if optional_ending_2:
                a2 = a2 or (j == 0)

        # Map to okay array
        if right_side:
            okay_idx = l - j
        else:
            okay_idx = j
        okay[okay_idx] = a2

        a2_old = a2
        a1_old = a1

    # Handle end of string constraint
    if guard.endOfString:
        if right_side:
            # Right guard at end of string: must match at position 0 (end of string)
            # Only position 0 (which represents matching from end) should be true
            for j in range(1, l + 1):
                okay[j] = False
        else:
            # Left guard at end of string: must match at position l (end of string)
            # Only position l should be true
            for j in range(l):
                okay[j] = False

    return _finish(okay)


def apply_rule(
    rule,
    morph,
    bank,
    allowed_morpheme=None,
    morphemes=None,
    spans_override=None,
    boundary_positions_override=None,
    order=None,
    word_to_slot_values=None,
    wid=None,
    return_trace=False,
    guard_cache=None,
    guard_cache_max=None,
):
    """Apply a phonological rule to a morph."""
    phonemes = morph.phonemes
    l = len(phonemes)

    focus = rule.focus
    structural_change = rule.structuralChange
    left_trigger = rule.leftTriggers
    right_trigger = rule.rightTriggers
    # Some callers construct Rule objects directly (e.g. unit tests, or pre-existing
    # global rules) without setting a `locality_mode` attribute. Default to global
    # behavior in that case.
    locality_mode = getattr(rule, "locality_mode", None)

    target_positions = None
    boundary_positions = None

    allowed_morpheme_span = None
    left_neighbor_span = None
    right_neighbor_span = None
    target_span = None
    spans = None  # maps (slot, value) -> (start, end) in the UR

    slot = None
    val = None
    if allowed_morpheme is not None:
        slot, val = allowed_morpheme

    # Optional trace for callers that want to track edits (e.g., to update
    # morpheme boundary positions across sequential rule applications).
    trace = {"op": "none", "old_len": l, "new_len": l}

    # First, reconstruct morpheme spans and all internal boundaries (unless overridden)
    vals = word_to_slot_values.get(wid, {})

    # Local rules should apply only to words that actually contain the
    # designated morpheme (slot, value). If the current surface word does
    # not have this morpheme, the rule is vacuous on this example.
    if slot is not None and val is not None and vals.get(slot) != val:
        return (morph, trace) if return_trace else morph
        
    if spans_override is not None:
        spans = spans_override
    else:
        pos = 0
        spans = {}
        boundaries = []
        for s in order:
            if s not in vals:
                continue
            key = (s, vals[s])
            part = morphemes.get(key, "")
            part_len = len(tokenize(part)) if part else 0
            start = pos
            end = pos + part_len
            spans[key] = (start, end)
            pos = end
            boundaries.append(end)

            # Only trust these boundaries if they line up with the current UR length
            if pos == l and boundaries:
                # Internal morpheme boundaries only (exclude final word boundary)
                boundary_positions = set(boundaries[:-1])

    if boundary_positions_override is not None:
        # Caller-provided boundary positions take precedence
        boundary_positions = set(
            p for p in boundary_positions_override if 0 <= p <= l
        )

    # Derive the allowed target positions based on the spans we just computed
    if allowed_morpheme is not None:
        if spans is None or allowed_morpheme not in spans:
            # If we cannot reliably locate the allowed morpheme span in the
            # current UR, do not attempt to apply a local rule.
            return (morph, trace) if return_trace else morph
        allowed_morpheme_span = spans[allowed_morpheme]

        # Find immediate neighbour slots (to the left and right) that are
        # actually present for this word
        present_slots_in_order = [s for s in order if s in vals]
        idx = present_slots_in_order.index(slot)
        if idx > 0:
            left_slot = present_slots_in_order[idx - 1]
            left_key = (left_slot, vals[left_slot])
            left_neighbor_span = spans.get(left_key)
        if 0 <= idx < len(present_slots_in_order) - 1:
            right_slot = present_slots_in_order[idx + 1]
            right_key = (right_slot, vals[right_slot])
            right_neighbor_span = spans.get(right_key)

        # Decide locality based on locality_mode and location of '+'
        left_specs = getattr(left_trigger, "specifications", [])
        right_specs = getattr(right_trigger, "specifications", [])
        left_has_boundary = any(
            isinstance(s, BoundarySpecification) for s in left_specs
        )
        right_has_boundary = any(
            isinstance(s, BoundarySpecification) for s in right_specs
        )

        # Restrict application either to the allowed morpheme itself,
        # or to one specific neighbor, depending on which guard contains
        # the boundary:
        #   - '+' in right guard -> allowed morpheme OR left neighbor
        #   - '+' in left guard  -> allowed morpheme OR right neighbor
        #   - '+' in both        -> allowed morpheme ONLY
        if locality_mode == "allowed_morpheme":
            target_span = allowed_morpheme_span
            target_positions = set(
                range(allowed_morpheme_span[0], allowed_morpheme_span[1])
            )
        else:  # locality_mode == "neighbor"
            target_positions = set()
            # '+' in right guard: use left neighbor
            if right_has_boundary and not left_has_boundary:
                target_span = left_neighbor_span
                if left_neighbor_span is not None:
                    target_positions.update(
                        range(left_neighbor_span[0], left_neighbor_span[1])
                    )
            # '+' in left guard: use right neighbor
            elif left_has_boundary and not right_has_boundary:
                target_span = right_neighbor_span
                if right_neighbor_span is not None:
                    target_positions.update(
                        range(right_neighbor_span[0], right_neighbor_span[1])
                    )
    else:
        target_positions = set(range(l))

    # Check if rule applies (at least one position where it triggers)
    right_okay = apply_guard_sophisticated(
        phonemes,
        right_trigger,
        True,
        bank,
        boundary_positions=boundary_positions,
        guard_cache=guard_cache,
        guard_cache_max=guard_cache_max,
    )
    left_okay = apply_guard_sophisticated(
        phonemes,
        left_trigger,
        False,
        bank,
        boundary_positions=boundary_positions,
        guard_cache=guard_cache,
        guard_cache_max=guard_cache_max,
    )

    # Handle metathesis rules (swap two adjacent segments).
    # We interpret the left guard as matching at the boundary immediately before
    # the first segment, and the right guard as matching at the boundary
    # immediately after the second segment.
    #
    # Special case (local, boundary-conditioned metathesis):
    # If either guard mentions a morpheme boundary '+', we require the swapped
    # pair to STRADDLE an internal morpheme boundary (i.e., boundary at j+1),
    # and we allow the swap even if it crosses the allowed morpheme edge.
    #
    # In this case we evaluate the guards *ignoring* the boundary specification
    # itself (since '+' is handled by the straddle requirement).
    if isinstance(structural_change, MetathesisSpecification):
        if not isinstance(focus, MetathesisFocus):
            return (morph, trace) if return_trace else morph

        left_has_boundary = any(
            isinstance(s, BoundarySpecification) for s in getattr(left_trigger, "specifications", [])
        )
        right_has_boundary = any(
            isinstance(s, BoundarySpecification) for s in getattr(right_trigger, "specifications", [])
        )
        boundary_conditioned = (left_has_boundary or right_has_boundary)

        # If boundary-conditioned, evaluate guards with BoundarySpecification stripped.
        if boundary_conditioned:
            left_specs_nb = [
                s for s in getattr(left_trigger, "specifications", []) if not isinstance(s, BoundarySpecification)
            ]
            right_specs_nb = [
                s for s in getattr(right_trigger, "specifications", []) if not isinstance(s, BoundarySpecification)
            ]
            left_guard_nb = Guard(
                getattr(left_trigger, "side", "L"),
                bool(getattr(left_trigger, "endOfString", False)),
                bool(getattr(left_trigger, "optionalEnding", False)),
                bool(getattr(left_trigger, "starred", False)),
                left_specs_nb,
            )
            right_guard_nb = Guard(
                getattr(right_trigger, "side", "R"),
                bool(getattr(right_trigger, "endOfString", False)),
                bool(getattr(right_trigger, "optionalEnding", False)),
                bool(getattr(right_trigger, "starred", False)),
                right_specs_nb,
            )
            left_okay_nb = apply_guard_sophisticated(
                phonemes,
                left_guard_nb,
                False,
                bank,
                boundary_positions=boundary_positions,
                guard_cache=guard_cache,
                guard_cache_max=guard_cache_max,
            )
            right_okay_nb = apply_guard_sophisticated(
                phonemes,
                right_guard_nb,
                True,
                bank,
                boundary_positions=boundary_positions,
                guard_cache=guard_cache,
                guard_cache_max=guard_cache_max,
            )
        else:
            left_okay_nb = left_okay
            right_okay_nb = right_okay

        triggered_pair = [False] * max(0, l - 1)
        for j in range(l - 1):
            if not (left_okay_nb[j] and right_okay_nb[j + 2]):
                continue
            if boundary_conditioned:
                # Require the swap to happen ACROSS a morpheme boundary.
                if boundary_positions is None or (j + 1) not in boundary_positions:
                    continue
            if not (
                matches_specification(phonemes[j], focus.s1)
                and matches_specification(phonemes[j + 1], focus.s2)
            ):
                continue
            if target_positions is not None:
                if j in target_positions and (j + 1) in target_positions:
                    pass
                else:
                    # Allow boundary-conditioned metathesis to straddle the allowed
                    # morpheme boundary (one segment inside, one in neighbor).
                    if (
                        boundary_conditioned
                        and allowed_morpheme_span is not None
                        and boundary_positions is not None
                    ):
                        a0, a1 = allowed_morpheme_span
                        ok_straddle = False
                        # straddle right edge of allowed morpheme at boundary position a1
                        if (j == a1 - 1) and (j + 1 == a1) and (a1 in boundary_positions):
                            ok_straddle = True
                        # straddle left edge of allowed morpheme at boundary position a0
                        if (j == a0 - 1) and (j + 1 == a0) and (a0 in boundary_positions):
                            ok_straddle = True
                        if not ok_straddle:
                            continue
                    else:
                        continue
            triggered_pair[j] = True

        out = []
        made_change = False
        j = 0
        while j < l:
            if j < l - 1 and triggered_pair[j]:
                out.append(phonemes[j + 1])
                out.append(phonemes[j])
                made_change = True
                j += 2
            else:
                out.append(phonemes[j])
                j += 1

        if not made_change:
            return (morph, trace) if return_trace else morph
        trace = {"op": "metathesis", "old_len": l, "new_len": l}
        out_m = Morph(out)
        return (out_m, trace) if return_trace else out_m
    
    # Check which positions match the focus
    middle_okay = []
    for j in range(l):
        if isinstance(focus, EmptySpecification):
            middle_okay.append(True)
        elif isinstance(focus, OffsetSpecification):
            middle_okay.append(phonemes[j] == phonemes[j + focus.offset])
        else:
            middle_okay.append(matches_specification(phonemes[j], focus))
    
    # Find triggered positions
    triggered = []
    for j in range(l):
        # For right guard, we check position j + 1
        right_idx = j + 1 if j + 1 < len(right_okay) else len(right_okay) - 1
        active = middle_okay[j] and right_okay[right_idx] and left_okay[j]
        if target_positions is not None and j not in target_positions:
            active = False
        triggered.append(active)
    
    # Handle deletion rules
    if isinstance(structural_change, EmptySpecification):
        # Check if rule actually applies
        rule_applies = False
        for j in range(l):
            if triggered[j]:
                if isinstance(focus, OffsetSpecification):
                    # Check if offset's index exists and phonemes match
                    if 0 <= j + focus.offset < l and phonemes[j] == phonemes[j + focus.offset]:
                        rule_applies = True
                        break
                else:
                    rule_applies = True
                    break
        if not rule_applies:
            return (morph, trace) if return_trace else morph
        # Delete triggered phonemes
        output = []
        deleted = []
        for j in range(l):
            should_delete = triggered[j]
            if isinstance(focus, OffsetSpecification):
                should_delete = should_delete and (0 <= j + focus.offset < l and phonemes[j] == phonemes[j + focus.offset])
            if not should_delete:
                output.append(phonemes[j])
            else:
                deleted.append(j)
        out_m = Morph(output)
        trace = {"op": "delete", "deleted": deleted, "old_len": l, "new_len": len(output)}
        return (out_m, trace) if return_trace else out_m
    
    # Handle insertion rules
    elif isinstance(focus, EmptySpecification):
        # Build output, inserting at a matching position
        output = []
        inserted = False
        # Find all valid insertion positions first
        valid_positions = []
        for j in range(l + 1):
            left_idx = j
            right_idx = j if j < len(right_okay) else len(right_okay) - 1
            # Check if guards match
            if not (left_okay[left_idx] and right_okay[right_idx]):
                continue
            # Respect locality
            if target_positions is not None:
                # Insertion position j corresponds to the gap before phoneme j;
                # it can plausibly be attributed either to the segment on the
                # left (j-1) or to the segment on the right (j). Using only
                # (j-1) incorrectly blocks insertions at the *start* of a
                # non-empty allowed morpheme (common for boundary-conditioned
                # copy-insertion rules like Ø -> -1 / + _).
                if l == 0:
                    left_anchor = 0
                    right_anchor = 0
                else:
                    left_anchor = max(0, j - 1)
                    right_anchor = min(j, l - 1)
                if left_anchor not in target_positions and right_anchor not in target_positions:
                    # Special case: the designated morpheme can be empty (span length 0),
                    # but we still want to allow insertions *at its boundary position*.
                    # Example: possessive="" + root="tʌt^s", boundary at j=0.
                    if (
                        target_span is not None
                        and target_span[0] == target_span[1]
                        and j == target_span[0]
                    ):
                        pass
                    else:
                        continue
            
            # For copy rules, we need to additionally check if the guard matches the segment we're copying from
            can_insert_here = True
            if isinstance(structural_change, OffsetSpecification):
                offset = structural_change.offset
                if offset < 0:
                    # Copy from left: check if left trigger matches the segment at j + offset
                    copy_idx = j + offset
                    if not (0 <= copy_idx < l):
                        can_insert_here = False  # Can't copy from out-of-bounds position
                    else:
                        # For copy rules, if the guard includes a concrete phoneme/feature
                        # spec in addition to '+', require the copied segment to satisfy it.
                        # BoundarySpecification itself should NOT be matched against segments.
                        left_spec = None
                        for s in getattr(left_trigger, "specifications", []) or []:
                            if not isinstance(s, BoundarySpecification):
                                left_spec = s
                                break
                        if left_spec is not None and not matches_specification(phonemes[copy_idx], left_spec):
                            can_insert_here = False  # Guard doesn't match copied segment
                elif offset > 0:
                    # Copy from right: check if right trigger matches the segment at j + offset - 1
                    copy_idx = j + offset - 1
                    if not (0 <= copy_idx < l):
                        can_insert_here = False  # Can't copy from out-of-bounds position
                    else:
                        right_spec = None
                        for s in getattr(right_trigger, "specifications", []) or []:
                            if not isinstance(s, BoundarySpecification):
                                right_spec = s
                                break
                        if right_spec is not None and not matches_specification(phonemes[copy_idx], right_spec):
                            can_insert_here = False  # Guard doesn't match copied segment
            
            if can_insert_here:
                valid_positions.append(j)
        
        # Choose insertion position.
        #
        # For copy rules, we want the insertion to occur at the boundary that the
        # '+' guard is referring to. In local mode, this is typically the *edge*
        # of the designated morpheme:
        #   - '+ _'  (boundary in LEFT guard): insert at start of allowed morpheme
        #   - '_ +'  (boundary in RIGHT guard): insert at end of allowed morpheme
        if isinstance(structural_change, OffsetSpecification) and valid_positions:
            if target_span is not None:
                left_has_boundary = any(
                    isinstance(s, BoundarySpecification)
                    for s in getattr(left_trigger, "specifications", []) or []
                )
                right_has_boundary = any(
                    isinstance(s, BoundarySpecification)
                    for s in getattr(right_trigger, "specifications", []) or []
                )
                preferred = None
                if left_has_boundary and not right_has_boundary:
                    preferred = target_span[0]
                elif right_has_boundary and not left_has_boundary:
                    preferred = target_span[1]
                # If '+' is present in exactly one guard, the intended semantics
                # is that insertion is anchored to that *specific* edge of the
                # allowed morpheme. If that edge is not a valid insertion site
                # (because a phoneme/feature condition fails), the rule should
                # simply not apply — it should NOT drift to a different morpheme
                # boundary later in the word.
                if preferred is not None:
                    j = preferred if preferred in valid_positions else None
                else:
                    # If both guards mention '+', or neither does, we cannot
                    # uniquely identify the intended boundary. Fall back to a
                    # conservative heuristic near the allowed span.
                    j = min(
                        valid_positions,
                        key=lambda pos: min(
                            abs(pos - target_span[0]), abs(pos - target_span[1])
                        ),
                    )
            else:
                # No span info: fall back to choosing the last valid position.
                j = valid_positions[-1]
        elif valid_positions:
            # Use first matching position for regular insertion
            j = valid_positions[0]
        else:
            j = None
        
        if j is not None:
            inserted = True
        else:
            inserted = False
        
        # Build output: add phonemes before insertion, then insert, then add phonemes after
        for i in range(l + 1):
            # Insert copied/constant segment at position j
            if i == j and inserted:
                if isinstance(structural_change, OffsetSpecification):
                    # Copying rule
                    offset = structural_change.offset
                    if offset > 0:
                        # Copy from right
                        copy_idx = j + offset - 1
                        if 0 <= copy_idx < l:
                            output.append(phonemes[copy_idx])
                    elif offset < 0:
                        # Copy from left
                        copy_idx = j + offset
                        if 0 <= copy_idx < l:
                            output.append(phonemes[copy_idx])
                elif isinstance(structural_change, ConstantPhoneme):
                    output.append(structural_change.p)
                elif isinstance(structural_change, FeatureMatrix):
                    # Feature matrix insertion - not typically used but handle it
                    pass
            
            # Add original phoneme
            if i < l:
                output.append(phonemes[i])
        
        if not inserted:
            return (morph, trace) if return_trace else morph
        
        out_m = Morph(output)
        trace = {"op": "insert", "index": j, "count": 1, "old_len": l, "new_len": len(output)}
        return (out_m, trace) if return_trace else out_m
    
    # Handle modification rules
    else:
        output = []
        made_change = False
        
        for j in range(l):
            if not triggered[j]:
                output.append(phonemes[j])
            else:
                # Apply structural change
                if isinstance(structural_change, OffsetSpecification):
                    # Copying
                    offset = structural_change.offset
                    copy_idx = j + offset
                    if 0 <= copy_idx < l:
                        output.append(phonemes[copy_idx])
                        if phonemes[copy_idx] != phonemes[j]:
                            made_change = True
                    else:
                        output.append(phonemes[j])
                elif isinstance(structural_change, PlaceSpecification):
                    # Place assimilation - simplified
                    output.append(phonemes[j])
                else:
                    # Feature change or constant
                    new_phoneme = apply_specification(phonemes[j], structural_change, bank)
                    output.append(new_phoneme)
                    if new_phoneme != phonemes[j]:
                        made_change = True
        
        if not made_change:
            return (morph, trace) if return_trace else morph
        
        out_m = Morph(output)
        trace = {"op": "modify", "old_len": l, "new_len": l}
        return (out_m, trace) if return_trace else out_m

