import itertools, math
from collections import defaultdict


class OrderScorer:
    """Helper class to score morpheme orders by how likely they are to be correct."""

    def __init__(self, surface_words, word_to_present_slots, slot_to_value_to_words):
        self.surface_words = surface_words
        self.word_to_present_slots = word_to_present_slots
        self.slot_to_value_to_words = slot_to_value_to_words

        self._slot_value_signatures = {}
        self._slot_value_mean_pos = {}
        self._slot_diff = defaultdict(float)
        self._word_substrings = {}
        self._substring_global_counts = defaultdict(int)
        self._prepare_slot_order_statistics()
    

    def score_order(self, order):
        """
        Score an order [slot_0, ..., slot_{n-1}] by:
            sum_{i < j} diff(slot_i, slot_j)
        where diff(slot_a, slot_b) is precomputed from mean signature positions.
        """
        if not self._slot_diff:
            return 0.0
        score = 0.0
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a = order[i]
                b = order[j]
                score += self._slot_diff.get((a, b), 0.0)
        return score
    
    
    def _prepare_slot_order_statistics(self):
        """Pre-compute signatures and diffs for order scoring."""
        self._slot_value_signatures.clear()
        self._slot_value_mean_pos.clear()
        self._slot_diff.clear()
        self._word_substrings.clear()
        self._substring_global_counts.clear()
        
        unique_words = sorted(set(self.surface_words))
        if not unique_words:
            return

        for word in unique_words:
            subs = self._enumerate_substrings(word)
            self._word_substrings[word] = subs
            for sub in subs:
                self._substring_global_counts[sub] += 1
        
        for slot, value_dict in self.slot_to_value_to_words.items():
            for value, words in value_dict.items():
                # `words` are word keys; signatures are over surface strings.
                word_list = list(dict.fromkeys([w[0] for w in words]))
                signature = self._select_signature_for_value(word_list)
                if not signature:
                    continue
                key = (slot, value)
                self._slot_value_signatures[key] = signature

        self._compute_slot_diffs()
    
    
    def _compute_slot_diffs(self):
        """
        Compute diff(slot_a, slot_b) as the mean difference in starting positions
        (slot_b - slot_a) across words containing both slot_a and slot_b.
        """
        all_slots = list(self.slot_to_value_to_words.keys())
        if not all_slots:
            return

        for slot_a, slot_b in itertools.permutations(all_slots, 2):
            values_a = list(self.slot_to_value_to_words.get(slot_a, {}).keys())
            values_b = list(self.slot_to_value_to_words.get(slot_b, {}).keys())
            if not values_a or not values_b:
                continue

            mean_pos_a_vals = []
            for val_a in values_a:
                key_a = (slot_a, val_a)
                sig_a = self._slot_value_signatures.get(key_a)
                if not sig_a:
                    continue
                words_a = self.slot_to_value_to_words.get(slot_a, {}).get(val_a, [])
                positions = []
                for w in words_a:
                    if slot_b not in self.word_to_present_slots.get(w, set()):
                        continue
                    p = w[0].find(sig_a)
                    if p != -1:
                        positions.append(p)
                if positions:
                    mean_pos_a_vals.append(sum(positions) / float(len(positions)))

            mean_pos_b_vals = []
            for val_b in values_b:
                key_b = (slot_b, val_b)
                sig_b = self._slot_value_signatures.get(key_b)
                if not sig_b:
                    continue
                words_b = self.slot_to_value_to_words.get(slot_b, {}).get(val_b, [])
                positions = []
                for w in words_b:
                    if slot_a not in self.word_to_present_slots.get(w, set()):
                        continue
                    p = w[0].find(sig_b)
                    if p != -1:
                        positions.append(p)
                if positions:
                    mean_pos_b_vals.append(sum(positions) / float(len(positions)))

            if not mean_pos_a_vals or not mean_pos_b_vals:
                continue
            self._slot_diff[(slot_a, slot_b)] = (
                (sum(mean_pos_b_vals) / float(len(mean_pos_b_vals)))
                - (sum(mean_pos_a_vals) / float(len(mean_pos_a_vals)))
            )
    

    def _select_signature_for_value(self, words):
        """
        Choose signature s* for a given (slot, value) from candidate strings s by:
            s* = argmax_s precision(s) * recall(s)
        where:
            precision(s) = (# words containing (slot, value) and s) / (# words containing (slot, value))
            recall(s)    = (# words containing (slot, value) and s) / (# words containing s)
        """
        if not words:
            return None

        n = len(words)
        sub_counts_in_value = defaultdict(int)
        for word in words:
            for sub in self._word_substrings.get(word, ()):
                sub_counts_in_value[sub] += 1

        best_sub = None
        best_score = -1.0
        for sub, count_in_value in sub_counts_in_value.items():
            if not sub:
                continue
            global_count = self._substring_global_counts.get(sub, 0)
            if global_count <= 0:
                continue
            precision = count_in_value / float(n)
            recall = count_in_value / float(global_count)
            score = precision * recall
            if (
                score > best_score
                or (
                    math.isclose(score, best_score)
                    and best_sub is not None
                    and len(sub) > len(best_sub)
                )
                or (math.isclose(score, best_score) and best_sub is None)
            ):
                best_sub = sub
                best_score = score
        return best_sub
    
    
    def _enumerate_substrings(self, word):
        """Enumerate all substrings of a word."""
        if not word:
            return set()
        max_len = len(word)
        subs = set()
        for length in range(1, max_len + 1):
            for start in range(0, len(word) - length + 1):
                subs.add(word[start:start + length])
        return subs