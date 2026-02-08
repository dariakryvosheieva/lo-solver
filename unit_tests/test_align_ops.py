import sys
from pathlib import Path

_bpl_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bpl_dir))

from features import FeatureBank
from rule_search import _align_ops


def normalize_ops(ops):
    return set(tuple(op) for op in ops)


def test_insertion_1():
    ur_tokens = ["p", "ɨ", "z", "a", "n", "e", "z", "ɨ", "t", "l", "ə", "n"]
    sr_tokens = ["p", "ɨ", "z", "a", "n", "n", "e", "z", "ɨ", "t", "l", "ə", "n"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["pɨzanetlə"]))
    expected = [
        [
            ("M", "p", "p", 0, 0),
            ("M", "ɨ", "ɨ", 1, 1),
            ("M", "z", "z", 2, 2),
            ("M", "a", "a", 3, 3),
            ("I", None, "n", 4, 4),
            ("M", "n", "n", 4, 5),
            ("M", "e", "e", 5, 6),
            ("M", "z", "z", 6, 7),
            ("M", "ɨ", "ɨ", 7, 8),
            ("M", "t", "t", 8, 9),
            ("M", "l", "l", 9, 10),
            ("M", "ə", "ə", 10, 11),
            ("M", "n", "n", 11, 12),
        ],
        [
            ("M", "p", "p", 0, 0),
            ("M", "ɨ", "ɨ", 1, 1),
            ("M", "z", "z", 2, 2),
            ("M", "a", "a", 3, 3),
            ("M", "n", "n", 4, 4),
            ("I", None, "n", 5, 5),
            ("M", "e", "e", 5, 6),
            ("M", "z", "z", 6, 7),
            ("M", "ɨ", "ɨ", 7, 8),
            ("M", "t", "t", 8, 9),
            ("M", "l", "l", 9, 10),
            ("M", "ə", "ə", 10, 11),
            ("M", "n", "n", 11, 12),
        ],
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_insertion_2():
    ur_tokens = ["h", "ɨ", "k", "a"]
    sr_tokens = ["ɨ", "n", "h", "ɨ", "k", "a"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["ɨnhka"]))
    expected = [
        [
            ("I", None, "ɨ", 0, 0),
            ("I", None, "n", 0, 1),
            ("M", "h", "h", 0, 2),
            ("M", "ɨ", "ɨ", 1, 3),
            ("M", "k", "k", 2, 4),
            ("M", "a", "a", 3, 5),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_deletion_1():
    ur_tokens = ["k", "u", "s", "a", "n", "a", "m", "a"]
    sr_tokens = ["k", "u", "s", "n", "a", "m", "a"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["kusnam"]))
    expected = [
        [
            ("M", "k", "k", 0, 0),
            ("M", "u", "u", 1, 1),
            ("M", "s", "s", 2, 2),
            ("D", "a", None, 3, 3),
            ("M", "n", "n", 4, 3),
            ("M", "a", "a", 5, 4),
            ("M", "m", "m", 6, 5),
            ("M", "a", "a", 7, 6),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_deletion_2():
    ur_tokens = ["q", "u", "c", "x", "a", "s", "a", "t^h", "a", "p", "s", "a"]
    sr_tokens = ["q", "u", "c", "x", "s", "t^h", "p", "s", "a"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["qucxast^hp"]))
    expected = [
        [
            ("M", "q", "q", 0, 0),
            ("M", "u", "u", 1, 1),
            ("M", "c", "c", 2, 2),
            ("M", "x", "x", 3, 3),
            ("D", "a", None, 4, 4),
            ("M", "s", "s", 5, 4),
            ("D", "a", None, 6, 5),
            ("M", "t^h", "t^h", 7, 5),
            ("D", "a", None, 8, 6),
            ("M", "p", "p", 9, 6),
            ("M", "s", "s", 10, 7),
            ("M", "a", "a", 11, 8),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_substitution_1():
    ur_tokens = ["n", "t^s", "a", "p"]
    sr_tokens = ["n", "d^z", "a", "p"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["nt^sd^zap"]))
    expected = [
        [
            ("M", "n", "n", 0, 0),
            ("S", "t^s", "d^z", 1, 1),
            ("M", "a", "a", 2, 2),
            ("M", "p", "p", 3, 3),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_substitution_2():
    ur_tokens = ["l", "u", "p", "a", "t", "o", "o", "s"]
    sr_tokens = ["l", "u", "v", "a", "r", "o", "o", "s"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["lupvatros"]))
    expected = [
        [
            ("M", "l", "l", 0, 0),
            ("M", "u", "u", 1, 1),
            ("S", "p", "v", 2, 2),
            ("M", "a", "a", 3, 3),
            ("S", "t", "r", 4, 4),
            ("M", "o", "o", 5, 5),
            ("M", "o", "o", 6, 6),
            ("M", "s", "s", 7, 7),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_integration_1():
    ur_tokens = ["k", "o", "m"]
    sr_tokens = ["ŋ", "g", "o", "m"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["ŋkgom"]))
    expected = [
        [
            ("I", None, "ŋ", 0, 0),
            ("S", "k", "g", 0, 1),
            ("M", "o", "o", 1, 2),
            ("M", "m", "m", 2, 3),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_integration_2():
    ur_tokens = ["h", "u", "l", "i"]
    sr_tokens = ["h", "ü", "l"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["huüli"]))
    expected = [
        [
            ("M", "h", "h", 0, 0),
            ("S", "u", "ü", 1, 1),
            ("M", "l", "l", 2, 2),
            ("D", "i", None, 3, 3),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


def test_metathesis():
    ur_tokens = ["p", "o", "k", "s", "k", "u", "y", "k", "ʌ", "s", "m", "ʌ"]
    sr_tokens = ["p", "o", "k", "s", "k", "u", "k", "y", "ʌ", "s", "m", "ʌ"]
    ops = _align_ops(ur_tokens, sr_tokens, bank=FeatureBank(["poksuyʌm"]))
    expected = [
        [
            ("M", "p", "p", 0, 0),
            ("M", "o", "o", 1, 1),
            ("M", "k", "k", 2, 2),
            ("M", "s", "s", 3, 3),
            ("M", "k", "k", 4, 4),
            ("M", "u", "u", 5, 5),
            ("T", ("y", "k"), ("k", "y"), 6, 6),
            ("M", "ʌ", "ʌ", 8, 8),
            ("M", "s", "s", 9, 9),
            ("M", "m", "m", 10, 10),
            ("M", "ʌ", "ʌ", 11, 11),
        ]
    ]
    assert normalize_ops(ops) == normalize_ops(expected)


if __name__ == "__main__":
    test_insertion_1()
    test_insertion_2()
    test_deletion_1()
    test_deletion_2()
    test_substitution_1()
    test_substitution_2()
    test_integration_1()
    test_integration_2()
    test_metathesis()
