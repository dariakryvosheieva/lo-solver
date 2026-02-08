class EditDistanceCalculator:
    """
    Helper class to compute feature edit distances.

    Feature edit distance is defined as the minimum cost of transforming one sequence of
    feature vectors into another:
    - insertions and deletions have fixed cost 0.5;
    - the cost of a substitution is the number of differing features divided by the total
      number of features;
    - metathesis has a small epsilon cost (to be cheaper than two substitutions) and is
      only allowed when the two phonemes are swapped exactly.
    """

    def __init__(self, bank, use_features=True):
        self.bank = bank
        self.use_features = use_features

        # caching to avoid repeated computations
        self._edit_distance_cache = {}
        self._feature_sequence_cache = {}
        self._feature_substitution_cache = {}

    def edit_distance(self, a, b):
        """Compute edit distance between two strings with caching."""
        if a <= b:
            key = (a, b)
        else:
            key = (b, a)
        if key in self._edit_distance_cache:
            return self._edit_distance_cache[key]

        dist = None
        if self.use_features:
            try:
                seq_a = self._get_feature_sequence(a)
                seq_b = self._get_feature_sequence(b)
                dist = self._feature_edit_distance(seq_a, seq_b)
            except Exception:
                dist = float("inf")
        else:
            dist = self._string_edit_distance(a, b)

        self._edit_distance_cache[key] = dist
        return dist

    def _get_feature_sequence(self, text):
        """Return a tuple of feature vectors (one per phoneme) for the given text."""
        if text in self._feature_sequence_cache:
            return self._feature_sequence_cache[text]
        if not text:
            self._feature_sequence_cache[text] = tuple()
            return tuple()

        matrices = self.bank.wordToMatrix(text)
        sequence = tuple(tuple(int(bool(bit)) for bit in matrix) for matrix in matrices)
        self._feature_sequence_cache[text] = sequence
        return sequence

    def _feature_substitution_cost(self, vec_a, vec_b):
        """Compute the cost of substituting one phoneme (=feature vector) for another."""
        if vec_a == vec_b:
            return 0.0
        key = (vec_a, vec_b) if vec_a <= vec_b else (vec_b, vec_a)
        if key in self._feature_substitution_cache:
            return self._feature_substitution_cache[key]
        diff = sum(1 for x, y in zip(vec_a, vec_b) if x != y)
        cost = diff / len(self.bank.features)
        self._feature_substitution_cache[key] = cost
        return cost

    def _feature_edit_distance(self, seq_a, seq_b):
        """Feature-based edit distance (recommended, default)."""
        m, n = len(seq_a), len(seq_b)

        indel_cost = 0.5

        if m == 0:
            return sum(indel_cost for _ in seq_b)
        if n == 0:
            return sum(indel_cost for _ in seq_a)

        epsilon = 1.0 / (len(self.bank.features) * 10)

        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + indel_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + indel_cost

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                delete_cost = dp[i - 1][j] + indel_cost
                insert_cost = dp[i][j - 1] + indel_cost
                sub_cost = dp[i - 1][j - 1] + self._feature_substitution_cost(
                    seq_a[i - 1], seq_b[j - 1]
                )
                best = min(delete_cost, insert_cost, sub_cost)
                # metathesis
                if i >= 2 and j >= 2:
                    a0, a1 = seq_a[i - 2], seq_a[i - 1]
                    b0, b1 = seq_b[j - 2], seq_b[j - 1]
                    if a0 != a1 and a0 == b1 and a1 == b0:
                        best = min(best, dp[i - 2][j - 2] + epsilon)
                dp[i][j] = best

        return dp[m][n]

    def _string_edit_distance(self, a, b):
        """Character-based edit distance (supported just in case, but not recommended)."""
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                best = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
                # Damerau-Levenshtein swap (metathesis)
                if (
                    i >= 2
                    and j >= 2
                    and a[i - 1] != a[i - 2]
                    and a[i - 2] == b[j - 1]
                    and a[i - 1] == b[j - 2]
                ):
                    best = min(best, dp[i - 2][j - 2] + 1)
                dp[i][j] = best
        return dp[m][n]

    def normalize_input(self, word):
        """Handle various input types."""
        if word is None:
            return ""
        if hasattr(word, "phonemes"):
            return "".join(word.phonemes)
        if isinstance(word, str):
            # Strip leading/trailing slashes that appear in Morph.__str__()
            if word.startswith("/") and word.endswith("/"):
                return word[1:-1]
            return word
        return str(word)
