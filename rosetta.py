import time
from collections import defaultdict

from rule import BoundarySpecification
from morpheme_search import MorphemeSolver
from utilities import parse, feature_dict_to_tuple


def _topo_order_from_constraints(all_roles, after):
	"""Return a deterministic topological order or None if cyclic."""
	indeg = {r: 0 for r in all_roles}
	for a in all_roles:
		for b in after[a]:
			indeg[b] += 1
	ready = sorted([r for r in all_roles if indeg[r] == 0])
	out = []
	while ready:
		r = ready.pop(0)
		out.append(r)
		for b in sorted(after[r]):
			indeg[b] -= 1
			if indeg[b] == 0:
				ready.append(b)
		ready.sort()
	return out if len(out) == len(all_roles) else None


def _path_exists(after, start, target):
	"""DFS reachability in the role constraint graph."""
	if start == target:
		return True
	seen = set()
	stack = [start]
	while stack:
		x = stack.pop()
		if x in seen:
			continue
		seen.add(x)
		for y in after[x]:
			if y == target:
				return True
			if y not in seen:
				stack.append(y)
	return False


def _add_role_edge(after, a, b):
	"""
	Add constraint a < b.
	Return:
		- True if edge newly added
		- False if already present
		- None if would create a cycle
	"""
	if a == b:
		return False
	if b in after[a]:
		return False
	if _path_exists(after, b, a):
		return None
	after[a].add(b)
	return True


def _longest_common_substring_length(a, b):
	"""Return the length of the longest common substring of a and b."""
	if not a or not b:
		return 0
	dp = [0] * (len(b) + 1)
	best = 0
	for i in range(1, len(a) + 1):
		new_dp = [0] * (len(b) + 1)
		for j in range(1, len(b) + 1):
			if a[i - 1] == b[j - 1]:
				new_dp[j] = dp[j - 1] + 1
				best = max(best, new_dp[j])
		dp = new_dp
	return best


class _MorphologyState:
	def __init__(self):
		self.assigned_by_pos = defaultdict(list)
		self.total_score = 0.0

	def clone(self):
		out = _MorphologyState()
		out.assigned_by_pos = defaultdict(
			list,
			{pos: list(assignments) for pos, assignments in self.assigned_by_pos.items()},
		)
		out.total_score = self.total_score
		return out

	def _pair_score(self, left, right):
		left_surface, left_feats = left
		right_surface, right_feats = right
		shared = set(left_feats) & set(right_feats)
		if not shared:
			return 0.0
		weight = 0
		for slot, _value in shared:
			weight += 3 if slot == "root" else 1
		return float(weight * _longest_common_substring_length(left_surface, right_surface))

	def estimate_total_with_additions(self, additions):
		if not additions:
			return self.total_score
		per_pos = defaultdict(list)
		for pos, surface, feats in additions:
			per_pos[pos].append((surface, feats))
		total = self.total_score
		for pos, extra in per_pos.items():
			current = self.assigned_by_pos[pos]
			for item in extra:
				for existing in current:
					total += self._pair_score(existing, item)
			for i in range(len(extra)):
				for j in range(i + 1, len(extra)):
					total += self._pair_score(extra[i], extra[j])
		return total

	def push(self, pos, surface, feats):
		item = (surface, feats)
		delta = 0.0
		for existing in self.assigned_by_pos[pos]:
			delta += self._pair_score(existing, item)
		self.assigned_by_pos[pos].append(item)
		self.total_score += delta
		return delta

	def pop(self, pos, delta):
		self.total_score -= delta
		self.assigned_by_pos[pos].pop()
		if not self.assigned_by_pos[pos]:
			del self.assigned_by_pos[pos]


def _estimate_sentence_total_score(s, t, english_to_problemese, morph_state):
	"""Best morphology heuristic obtainable for assigning translation t to sentence s."""
	n = len(s)
	used_positions = set()
	additions = []
	best = float("-inf")

	def candidates_for_token(tok_i):
		base, feats, pos, _ = t[tok_i]
		key = (base, feats, pos)
		if key in english_to_problemese:
			w = english_to_problemese[key]
			return [p for p in range(n) if p not in used_positions and s[p] == w]
		return [p for p in range(n) if p not in used_positions]

	def pick_next_token():
		best_i = None
		best_cands = None
		for i in range(n):
			if any(idx == i for idx, _ in additions):
				continue
			cands = candidates_for_token(i)
			if not cands:
				return None, None
			if best_cands is None or len(cands) < len(best_cands):
				best_i, best_cands = i, cands
		return best_i, best_cands

	def rec():
		nonlocal best
		if len(additions) == n:
			best = max(best, morph_state.estimate_total_with_additions(
				[(pos, surface, feats) for _, (pos, surface, feats) in additions]
			))
			return

		i, cands = pick_next_token()
		if i is None:
			return

		_, feats, pos, _ = t[i]
		scored = []
		for p in cands:
			score = morph_state.estimate_total_with_additions([(pos, s[p], feats)])
			scored.append((score, p))
		for _, p in sorted(scored, reverse=True):
			used_positions.add(p)
			additions.append((i, (pos, s[p], feats)))
			rec()
			additions.pop()
			used_positions.remove(p)

	rec()
	return best


def _solve_tokens_in_sentence(
	s,
	t,
	english_to_problemese,
	after,
	continue_fn,
	on_assign_token=None,
	prune_fn=None,
	position_sort_key=None,
):
	"""
	Backtracking token-to-position assignment for a single sentence.
	Mutates (and backtracks) `english_to_problemese` and `after` in-place.
	"""
	n = len(s)
	used_positions = set()
	assigned_pos = [None] * n  # per translation token index

	def candidates_for_token(tok_i):
		(base, feats, pos, _) = t[tok_i]
		key = (base, feats, pos)
		if key in english_to_problemese:
			w = english_to_problemese[key]
			return [p for p in range(n) if p not in used_positions and s[p] == w]
		return [p for p in range(n) if p not in used_positions]

	def pick_next_token():
		best_i = None
		best_cands = None
		for i in range(n):
			if assigned_pos[i] is not None:
				continue
			cands = candidates_for_token(i)
			if not cands:
				return None, None
			if best_cands is None or len(cands) < len(best_cands):
				best_i, best_cands = i, cands
				if len(best_cands) == 1:
					break
		return best_i, best_cands

	def assign_tokens():
		if all(p is not None for p in assigned_pos):
			return continue_fn()

		i, cands = pick_next_token()
		if i is None:
			return None

		(base, feats, pos, role_i) = t[i]
		key = (base, feats, pos)
		if position_sort_key is not None:
			cands = sorted(cands, key=lambda p: position_sort_key(i, p))

		for p in cands:
			undo = []

			# mapping update (if needed)
			if key not in english_to_problemese:
				english_to_problemese[key] = s[p]
				undo.append(("map", key))

			# assign position
			assigned_pos[i] = p
			used_positions.add(p)
			undo.append(("pos", p, i))

			if on_assign_token is not None:
				extra_undo = on_assign_token(s[p], t[i])
				if extra_undo is not None:
					undo.append(("extra", extra_undo))

			# role-order constraints induced by relative order to already-assigned tokens
			ok = True
			for j in range(n):
				if j == i or assigned_pos[j] is None:
					continue
				role_j = t[j][3]
				if role_j == role_i:
					continue
				if assigned_pos[j] < p:
					a, b = role_j, role_i
				else:
					a, b = role_i, role_j
				res = _add_role_edge(after, a, b)
				if res is None:
					ok = False
					break
				if res is True:
					undo.append(("edge", a, b))

			if ok and not (prune_fn is not None and prune_fn()):
				res = assign_tokens()
				if res is not None:
					return res

			# undo
			for tag, *payload in reversed(undo):
				if tag == "edge":
					a, b = payload
					after[a].remove(b)
				elif tag == "extra":
					(fn,) = payload
					fn()
				elif tag == "pos":
					pp, ii = payload
					used_positions.remove(pp)
					assigned_pos[ii] = None
				elif tag == "map":
					(k,) = payload
					del english_to_problemese[k]

		return None

	return assign_tokens()


def _solve_word_order_and_lexicon(
	sentences,
	translations,
	roles,
	scrambled,
	sentence_word_sets=None,
	max_complete_candidates=1,
	beam_width=1000,
	debug=False,
):
	"""
	- if scrambled=True, assign each Problemese sentence to an English translation
	- assign each translation token to a position in its sentence
	- induce a consistent mapping (base, feats, pos) -> problemese_word
	- induce a global role order as a total order extending pairwise constraints
	"""
	english_to_problemese = {}
	after = {r: set() for r in roles}  # directed edges r -> roles that must come after r

	if scrambled:
		if len(sentences) != len(translations):
			raise ValueError(
				f"Scrambled rosetta requires equal counts: {len(sentences)} sentences vs {len(translations)} translations"
			)

		complete_candidates_collected = 0
		expanded_states = 0

		translations_by_len = defaultdict(list)
		for ti, t in enumerate(translations):
			translations_by_len[len(t)].append(ti)

		def copy_after_map(after_map):
			return {r: set(vs) for r, vs in after_map.items()}

		def state_signature(state):
			return (
				tuple(state["holes"]),
				tuple(sorted(state["english_to_problemese"].items())),
				tuple(
					(r, tuple(sorted(state["after"][r])))
					for r in sorted(state["after"])
				),
				tuple(
					(pos, tuple(sorted(assignments)))
					for pos, assignments in sorted(state["morph_state"].assigned_by_pos.items())
				),
			)

		def translation_candidates_for_sentence(state, sent_idx):
			s = sentences[sent_idx]
			sset = sentence_word_sets[sent_idx]
			english_to_problemese = state["english_to_problemese"]
			used_translations = state["used_translations"]
			morph_state = state["morph_state"]
			cands = []
			for ti in translations_by_len.get(len(s), []):
				if ti in used_translations:
					continue
				t = translations[ti]
				ok = True
				for (base, feats, pos, _) in t:
					key = (base, feats, pos)
					if key in english_to_problemese and english_to_problemese[key] not in sset:
						ok = False
						break
				if ok:
					estimate = _estimate_sentence_total_score(
						s,
						t,
						english_to_problemese,
						morph_state,
					)
					cands.append((estimate, ti))
			cands.sort(reverse=True)
			return cands

		def pick_next_sentence(state):
			holes = state["holes"]
			best_si = None
			best_cands = None
			best_gap = None
			best_top_score = None
			for si in range(len(sentences)):
				if holes[si] is not None:
					continue
				cands = translation_candidates_for_sentence(state, si)
				if not cands:
					return None, None, None, None
				top_score = cands[0][0]
				gap = float("inf") if len(cands) == 1 else (cands[0][0] - cands[1][0])
				if (
					best_cands is None
					or len(cands) < len(best_cands)
					or (
						len(cands) == len(best_cands)
						and (gap > best_gap or (gap == best_gap and top_score > best_top_score))
					)
				):
					best_si, best_cands = si, cands
					best_gap = gap
					best_top_score = top_score
					if len(best_cands) == 1:
						break
			return best_si, best_cands, best_gap, best_top_score

		def enumerate_sentence_extensions(state, sent_idx, ti):
			s = sentences[sent_idx]
			t = translations[ti]
			holes = list(state["holes"])
			holes[sent_idx] = ti
			used_translations = set(state["used_translations"])
			used_translations.add(ti)
			english_to_problemese = dict(state["english_to_problemese"])
			after = copy_after_map(state["after"])
			morph_state = state["morph_state"].clone()
			results = []

			def on_assign_token(surface, token):
				_, feats, pos, _ = token
				delta = morph_state.push(pos, surface, feats)
				return lambda: morph_state.pop(pos, delta)

			def position_sort_key(tok_i, p):
				_, feats, pos, _ = t[tok_i]
				return -morph_state.estimate_total_with_additions([(pos, s[p], feats)])

			def capture_state():
				results.append(
					{
						"holes": list(holes),
						"used_translations": set(used_translations),
						"english_to_problemese": dict(english_to_problemese),
						"after": copy_after_map(after),
						"morph_state": morph_state.clone(),
					}
				)
				return None

			_solve_tokens_in_sentence(
				s,
				t,
				english_to_problemese,
				after,
				capture_state,
				on_assign_token=on_assign_token,
				position_sort_key=position_sort_key,
			)
			return results

		if debug:
			print(f"starting word search for {len(sentences)} sentences / {len(translations)} translations")

		initial_state = {
			"holes": [None] * len(sentences),
			"used_translations": set(),
			"english_to_problemese": {},
			"after": copy_after_map(after),
			"morph_state": _MorphologyState(),
		}
		frontier = [initial_state]

		for depth in range(len(sentences)):
			if not frontier:
				break
			if debug:
				print(f"beam layer {depth}/{len(sentences)} frontier={len(frontier)} expanded={expanded_states}")
			next_states = []
			for state in frontier:
				expanded_states += 1
				sent_idx, trans_cands, _, _ = pick_next_sentence(state)
				if sent_idx is None:
					continue
				for _estimate, ti in trans_cands:
					next_states.extend(enumerate_sentence_extensions(state, sent_idx, ti))

			if not next_states:
				frontier = []
				break

			deduped = {}
			for state in next_states:
				key = state_signature(state)
				score = state["morph_state"].total_score
				prev = deduped.get(key)
				if prev is None or score > prev["morph_state"].total_score:
					deduped[key] = state

			frontier = sorted(
				deduped.values(),
				key=lambda st: (
					-st["morph_state"].total_score,
					-sum(h is not None for h in st["holes"]),
				),
			)[:beam_width]
			if debug:
				print(f"beam layer {depth + 1} kept {len(frontier)} states; " + \
					  f"best heuristic={frontier[0]['morph_state'].total_score:.2f}")

		complete_candidates = []
		for state in frontier:
			if not all(h is not None for h in state["holes"]):
				continue
			order = _topo_order_from_constraints(roles, state["after"])
			if order is None:
				continue
			complete_candidates.append(
				(
					state["morph_state"].total_score,
					list(state["holes"]),
					order,
					dict(state["english_to_problemese"]),
					{
						pos: tuple(sorted(assignments))
						for pos, assignments in state["morph_state"].assigned_by_pos.items()
					},
				)
			)

		complete_candidates.sort(key=lambda x: x[0], reverse=True)
		complete_candidates = complete_candidates[:max_complete_candidates]
		complete_candidates_collected = len(complete_candidates)

		if complete_candidates:
			if debug:
				print(f"word search finished: complete_collected={complete_candidates_collected}, expanded={expanded_states}")
			return complete_candidates
		else:
			if debug:
				print(f"word search failed after expanded={expanded_states}, complete_collected={complete_candidates_collected}")
			return None

	# Non-scrambled: translation i is paired with sentence i
	if len(sentences) != len(translations):
		raise ValueError(
			f"Rosetta requires equal counts: {len(sentences)} sentences vs {len(translations)} translations"
		)
	for idx, (s, t) in enumerate(zip(sentences, translations)):
		if len(s) != len(t):
			raise ValueError(f"Sentence/translation length mismatch for sentence {idx}")

	def solve_sentence(sent_idx):
		if sent_idx >= len(sentences):
			order = _topo_order_from_constraints(roles, after)
			if order is None:
				return None
			return order, dict(english_to_problemese)
		return _solve_tokens_in_sentence(
			sentences[sent_idx],
			translations[sent_idx],
			english_to_problemese,
			after,
			lambda: solve_sentence(sent_idx + 1),
		)

	return solve_sentence(0)


def solve_morphemes_and_rules(translations, english_to_problemese, debug=False):
	"""
	Solve morphemes and phonological rules for a fixed alignment of English and Problemese words.
	"""
	bf_to_pos = defaultdict(set)
	for t in translations:
		for base, feats, pos, _ in t:
			bf_to_pos[(base, feats)].add(pos)
	all_pos = sorted({pos for t in translations for (_, _, pos, _) in t})

	morpheme_orders = {}
	lexicon_union = set()
	global_rules_union = set()
	local_rules_union = set()

	for pos in all_pos:
		english_words_pos = sorted({
			(b, f)
			for (b, f), pos_set in bf_to_pos.items()
			if pos in pos_set
		})
		if not english_words_pos:
			continue
		word_features = [f for (_, f) in english_words_pos]
		word_features_no_root = sorted(set(f[1:] for f in word_features))
		base_list = sorted({b for (b, _) in english_words_pos})

		matrix_rows = []
		for b in base_list:
			row = []
			for feats_no_root in word_features_no_root:
				fmap = dict(feats_no_root)
				fmap["root"] = b
				f = feature_dict_to_tuple(fmap)
				row.append(english_to_problemese.get((b, f, pos), None))
			matrix_rows.append(tuple(row))

		res = MorphemeSolver(
			matrix_rows,
			word_features,
			max_total_cost=35,
			max_rule_cost=10,
			k=1,
			allow_empty_slots=True,
			segmentations_beyond_minimal=0,
			segmentation_queue_limit=100000,
			segmentation_queue_trim_factor=1.5,
			initial_lexicon_max_states=float("inf"),
			debug=debug,
		).solve()
		if res is None:
			print(f"Morpheme search failed for POS {pos}")
			return None
		morpheme_order, lexicon, global_rules, local_rules = res
		morpheme_orders[pos] = morpheme_order
		lexicon_union.update(lexicon)
		global_rules_union.update(global_rules)
		local_rules_union.update(local_rules)

	return (morpheme_orders, sorted(lexicon_union), global_rules_union, local_rules_union)


def solve_rosetta(pairs, scrambled=False, debug=False):
	"""
	Solve a 'Rosetta Stone' problem.

    Input:
	- scrambled=False (default): list of (sentence, translation)
		- sentence: raw string in Problemese (= target language)
		- translation: list of (features, pos, role) tuples representing words
			- features: dict of morphological features (e.g., {'root': 'problem', 'number': 'pl'})
			- pos: part of speech (e.g., 'N' for noun)
			- role: syntactic role (e.g., 'S' for subject)
	- scrambled=True: (list_of_sentences, list_of_translations)
    See problems_easy.py and problems_hard.py for example inputs.

    Output:
	- scrambled=False: (word_order, morpheme_orders, lexicon, global_rules, local_rules)
	- scrambled=True: (correspondences, word_order, morpheme_orders, lexicon, global_rules, local_rules)
	where:
	- correspondences: list of indices mapping Problemese sentences to English translations
	- word_order: list of roles by position (e.g., ['S','V','O'])
    - morpheme_orders: dict mapping each POS to a list of feature slots by position (e.g., {'N': ['root', 'number']})
    - lexicon: set of (underlying form, meaning) for each morpheme (e.g., {('problēma', 'problem'), ('ta', 'pl')})
    - global_rules: set of Rule objects - true phonological rules that apply whenever they can
    - local_rules: set of ((feature_slot, feature_value), Rule) - rules explaining allomorphic alternations of specific morphemes
	"""
	### Process input data ###
	translations = []
	roles = set()
	if scrambled:
		raw_sentences, raw_translations = pairs
		sentences = [parse(s) for s in raw_sentences]
		for translation in raw_translations:
			t = [
				(features["root"], feature_dict_to_tuple(features), pos, role)
				for features, pos, role in translation
			]
			translations.append(t)
			roles.update(role for (_, _, _, role) in t)
	else:
		sentences = []
		for sentence, translation in pairs:
			sentences.append(parse(sentence))
			translations.append([
				(features["root"], feature_dict_to_tuple(features), pos, role)
				for features, pos, role in translation
			])
			roles.update(role for _, _, role in translation)
	roles = sorted(roles)
	sentence_word_sets = [set(s) for s in sentences]

	### Solve word order and English-to-Problemese word assignment ###
	word_solve = _solve_word_order_and_lexicon(
		sentences,
		translations,
		roles,
		scrambled=scrambled,
		sentence_word_sets=sentence_word_sets,
		max_complete_candidates=1,
		beam_width=1000,
		debug=debug,
	)
	if word_solve is None:
		print("Word search failed")
		if scrambled:
			return [], [], {}, [], set(), set()
		return [], {}, [], set(), set()

	### Solve morphemes and phonological rules ###
	if scrambled:
		for i, candidate in enumerate(word_solve):
			heuristic_score, correspondences, word_order, english_to_problemese, _ = candidate
			if debug:
				print(f"Trying candidate {i + 1} (heuristic_score={heuristic_score:.2f}, correspondences={correspondences})")
			aligned_translations = [
				translations[correspondences[i]]
				for i in range(len(sentences))
			]
			ret = solve_morphemes_and_rules(aligned_translations, english_to_problemese, debug=debug)
			if ret is not None:
				return (correspondences, word_order, *ret)
		print("No valid solution found")
		return [], [], {}, [], set(), set()
	else:
		word_order, english_to_problemese = word_solve
		ret = solve_morphemes_and_rules(translations, english_to_problemese, debug=debug)
		if ret is not None:
			return (word_order, *ret)
		return [], {}, [], set(), set()


def format_solution_rosetta(problem, scrambled=False, debug=False):
	start_time = time.time()
	if scrambled:
		correspondences, word_order, morpheme_orders, lexicon, global_rules, local_rules = solve_rosetta(
			problem,
			scrambled=True,
			debug=debug
		)
		print("Correspondences:")
		for i, c in enumerate(correspondences):
			print(f"  ({i + 1}. {chr(65 + c)})")
	else:
		word_order, morpheme_orders, lexicon, global_rules, local_rules = solve_rosetta(
			problem,
			scrambled=False,
			debug=debug
		)
	if len(word_order) > 1:
		print("Word order:", " - ".join(r for r in word_order))
	for pos in morpheme_orders.keys():
		slots = [slot for slot in morpheme_orders[pos]]
		if len(slots) > 1:
			print(f"Morpheme order ({pos}):", " - ".join(slots))
	print("Vocabulary:")
	for w, m in lexicon:
		print(f"  ({w}, {m})")
	if global_rules or local_rules:
		print("Rules:")
		if global_rules:
			for r in global_rules:
				print(r.pretty())
		if local_rules:
			for r in local_rules:
				if isinstance(r, tuple):
					k, v = r[0]
					locality_mode = r[1].locality_mode
					if locality_mode == "allowed_morpheme":
						pretty_locality_mode = ""
					elif any(
						isinstance(s, BoundarySpecification)
						for s in r[1].leftTriggers.specifications
					):
						pretty_locality_mode = "right neighbor of "
					elif any(
						isinstance(s, BoundarySpecification)
						for s in r[1].rightTriggers.specifications
					):
						pretty_locality_mode = "left neighbor of "
					print(f"{pretty_locality_mode}({k}, {v}): {r[1].pretty()}")
	print(f"Time taken: {time.time() - start_time:.2f} s")
