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


def _solve_tokens_in_sentence(s, t, english_to_problemese, after, continue_fn):
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

			if ok:
				res = assign_tokens()
				if res is not None:
					return res

			# undo
			for tag, *payload in reversed(undo):
				if tag == "edge":
					a, b = payload
					after[a].remove(b)
				elif tag == "pos":
					pp, ii = payload
					used_positions.remove(pp)
					assigned_pos[ii] = None
				elif tag == "map":
					(k,) = payload
					del english_to_problemese[k]

		return None

	return assign_tokens()


def _solve_word_order_and_lexicon(sentences, translations, roles, scrambled, sentence_word_sets=None):
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

		holes = [None] * len(sentences)  # sentence index -> translation index
		used_translations = set()

		translations_by_len = defaultdict(list)
		for ti, t in enumerate(translations):
			translations_by_len[len(t)].append(ti)

		def translation_candidates_for_sentence(sent_idx):
			s = sentences[sent_idx]
			sset = sentence_word_sets[sent_idx]
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
					cands.append(ti)
			return cands

		def pick_next_sentence():
			best_si = None
			best_cands = None
			for si in range(len(sentences)):
				if holes[si] is not None:
					continue
				cands = translation_candidates_for_sentence(si)
				if not cands:
					return None, None
				if best_cands is None or len(cands) < len(best_cands):
					best_si, best_cands = si, cands
					if len(best_cands) == 1:
						break
			return best_si, best_cands

		def solve_next():
			if all(h is not None for h in holes):
				order = _topo_order_from_constraints(roles, after)
				if order is None:
					return None
				return list(holes), order, dict(english_to_problemese)

			sent_idx, trans_cands = pick_next_sentence()
			if sent_idx is None:
				return None

			s = sentences[sent_idx]
			for ti in trans_cands:
				holes[sent_idx] = ti
				used_translations.add(ti)
				t = translations[ti]

				res = _solve_tokens_in_sentence(s, t, english_to_problemese, after, solve_next)
				if res is not None:
					return res

				holes[sent_idx] = None
				used_translations.remove(ti)
			return None

		return solve_next()

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
			return None, order, dict(english_to_problemese)
		return _solve_tokens_in_sentence(
			sentences[sent_idx],
			translations[sent_idx],
			english_to_problemese,
			after,
			lambda: solve_sentence(sent_idx + 1),
		)

	return solve_sentence(0)


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
	)
	if word_solve is None:
		print("Word search failed")
		if scrambled:
			return [], [], {}, [], set(), set()
		return [], {}, [], set(), set()

	### Solve morphemes and phonological rules ###
	correspondences, word_order, english_to_problemese = word_solve
	aligned_translations = (
		[translations[correspondences[i]] for i in range(len(sentences))]
		if scrambled
		else translations
	)

	bf_to_pos = defaultdict(set)
	for t in aligned_translations:
		for base, feats, pos, _ in t:
			bf_to_pos[(base, feats)].add(pos)
	all_pos = sorted({pos for t in aligned_translations for (_, _, pos, _) in t})

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
			continue
		morpheme_order, lexicon, global_rules, local_rules = res
		morpheme_orders[pos] = morpheme_order
		lexicon_union.update(lexicon)
		global_rules_union.update(global_rules)
		local_rules_union.update(local_rules)

	### Return results ###
	ret = (word_order, morpheme_orders, sorted(lexicon_union), global_rules_union, local_rules_union)
	if scrambled:
		ret = (correspondences, *ret)
	return ret


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
