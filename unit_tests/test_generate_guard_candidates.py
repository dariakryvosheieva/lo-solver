import sys
from pathlib import Path

_bpl_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bpl_dir))

from rule_search import RuleSolver
from features import FeatureBank
from morph import Morph
from rule import (
    MetathesisFocus,
    MetathesisSpecification,
    EmptySpecification,
    OffsetSpecification,
    ConstantPhoneme,
    FeatureMatrix,
)


def normalize_candidates(candidates):
    return [(str(l), str(r)) for l, r in candidates]


def test_insertion_1():
    wid1 = ("ŋkom", (("root", "post"), ("poss", "poss")))
    wid2 = ("nt^sap", (("root", "sky"), ("poss", "poss")))
    wid3 = ("nt^say", (("root", "vine"), ("poss", "poss")))
    wid4 = ("ntʌt^s", (("root", "tooth"), ("poss", "poss")))
    examples = [
        (
            Morph("kom"),
            Morph("ŋkom"),
            wid1,
        ),
        (
            Morph("t^sap"),
            Morph("nt^sap"),
            wid2,
        ),
        (
            Morph("t^say"),
            Morph("nt^say"),
            wid3,
        ),
        (
            Morph("tʌt^s"),
            Morph("ntʌt^s"),
            wid4,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "post", "poss": "poss"},
        wid2: {"root": "sky", "poss": "poss"},
        wid3: {"root": "vine", "poss": "poss"},
        wid4: {"root": "tooth", "poss": "poss"},
    }
    segmentation = {
        wid1: [0, 1],
        wid2: [0, 1],
        wid3: [0, 1],
        wid4: [0, 1],
    }
    order = ["poss", "root"]
    morphemes = {
        ("poss", "poss"): "",
        ("root", "post"): "kom",
        ("root", "sky"): "t^sap",
        ("root", "vine"): "t^say",
        ("root", "tooth"): "tʌt^s",
    }
    solver = RuleSolver(
        examples=examples,
        bank=FeatureBank(["ŋkomnt^sapytʌ"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )
    candidates_ŋ = solver._generate_guard_candidates(
        float("inf"), EmptySpecification(), ConstantPhoneme("ŋ"), freq_cap=10
    )
    candidates_n = solver._generate_guard_candidates(
        float("inf"), EmptySpecification(), ConstantPhoneme("n"), freq_cap=10
    )
    assert normalize_candidates(candidates_ŋ) == [
        ("", ""),
        ("", "k"),
        ("", "[ +velar ]"),
        ("", "[ -nasal ]"),
        ("", "[ -voice ]"),
        ("", "[ -sonorant ]"),
        ("", "[ +stop ]"),
        ("", "[ -middle ]"),
        ("", "[ -tense ]"),
        ("", "[ -back ]"),
    ]
    assert normalize_candidates(candidates_n) == [
        ("", ""),
        ("", "[ -velar ]"),
        ("", "[ -nasal ]"),
        ("", "[ -voice ]"),
        ("", "[ -sonorant ]"),
        ("", "[ -middle ]"),
        ("", "[ -tense ]"),
        ("", "[ -back ]"),
        ("", "[ -rounded ]"),
        ("", "[ -vowel ]"),
    ]


def test_insertion_2():
    wid1 = (
        "pɨzannezɨtlən",
        (
            ("root", "desk"),
            ("case", "of"),
            ("number", "pl"),
            ("poss", "your-sg"),
        ),
    )
    wid2 = (
        "juɕɕezə",
        (
            ("root", "swan"),
            ("number", "pl"),
            ("poss", "my"),
        ),
    )
    wid3 = (
        "vərrezlən",
        (
            ("root", "forest"),
            ("case", "of"),
            ("number", "pl"),
        ),
    )
    wid4 = (
        "kəinnezɨs",
        (
            ("root", "wolf"),
            ("number", "pl"),
            ("poss", "his"),
        ),
    )
    examples = [
        (
            Morph("pɨzanezɨtlən"),
            Morph("pɨzannezɨtlən"),
            wid1,
        ),
        (
            Morph("juɕezə"),
            Morph("juɕɕezə"),
            wid2,
        ),
        (
            Morph("vərezlən"),
            Morph("vərrezlən"),
            wid3,
        ),
        (
            Morph("kəinezɨs"),
            Morph("kəinnezɨs"),
            wid4,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "desk", "case": "of", "number": "pl", "poss": "your-sg"},
        wid2: {"root": "swan", "number": "pl", "poss": "my"},
        wid3: {"root": "forest", "case": "of", "number": "pl"},
        wid4: {"root": "wolf", "number": "pl", "poss": "his"},
    }
    segmentation = {
        wid1: [0, 6, 8, 10],
        wid2: [0, 4, 6, 7],
        wid3: [0, 4, 6, 6],
        wid4: [0, 5, 7, 9],
    }
    order = ["root", "number", "poss", "case"]
    morphemes = {
        ("root", "desk"): "pɨzan",
        ("root", "swan"): "juɕ",
        ("root", "forest"): "vər",
        ("root", "wolf"): "kəin",
        ("root", "lake"): "tɨ",
        ("root", "house"): "cerku",
        ("number", "pl"): "ez",
        ("poss", "my"): "ə",
        ("poss", "your-sg"): "ɨt",
        ("poss", "his"): "ɨs",
        ("case", "of"): "lən",
        ("case", "from"): "liɕ",
        ("case", "with"): "kət",
    }
    candidates = RuleSolver(
        examples=examples,
        bank=FeatureBank(["pɨzanetləjuɕvrkis"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        allowed_morpheme=("number", "pl"),
        morphemes=morphemes,
    )._generate_guard_candidates(
        float("inf"), EmptySpecification(), OffsetSpecification(-1), freq_cap=10
    )
    assert normalize_candidates(candidates) == [
        ("", "+"),
        ("", "+ e"),
        ("", "+ [ -bilabial ]"),
        ("", "+ [ -stop ]"),
        ("", "+ [ +voice ]"),
        ("", "+ [ +tense ]"),
        ("", "+ [ -high ]"),
        ("", "+ [ -back ]"),
        ("", "+ [ -central ]"),
        ("", "+ [ +vowel ]"),
    ]


def test_deletion_1():
    wid1 = ("kusnama", (("root", "tongue"), ("poss", "cat")))
    examples = [
        (
            Morph("kusanama"),
            Morph("kusnama"),
            wid1,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "tongue", "poss": "cat"},
    }
    segmentation = {
        wid1: [0, 3],
    }
    order = ["poss", "root"]
    morphemes = {
        ("poss", "cat"): "kusa",
        ("root", "tongue"): "nama",
    }
    candidates = RuleSolver(
        examples=examples,
        bank=FeatureBank(["kusnam"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._generate_guard_candidates(
        float("inf"),
        FeatureMatrix([(True, "vowel")]),
        EmptySpecification(),
        freq_cap=18,
    )
    assert normalize_candidates(candidates) == [
        ("", ""),
        ("", "n"),
        ("", "[ -velar ]"),
        ("", "[ -stop ]"),
        ("", "[ +voice ]"),
        ("", "[ -tense ]"),
        ("", "[ -high ]"),
        ("", "[ -back ]"),
        ("", "[ -rounded ]"),
        ("", "[ -vowel ]"),
        ("", "[ +sonorant ]"),
        ("", "[ -fricative ]"),
        ("", "[ +alveolar ]"),
        ("", "[ +nasal ]"),
        ("", "[ -low ]"),
        ("", "[ -central ]"),
        ("", "[ -bilabial ]"),
        ("", "+"),
    ]


def test_deletion_2():
    wid1 = ("utŋa", (("root", "house"), ("poss", "my")))
    wid2 = ("čušp^ha", (("root", "corner"), ("poss", "his")))
    wid3 = ("išma", (("root", "bed"), ("poss", "your")))
    examples = [
        (
            Morph("utaŋa"),
            Morph("utŋa"),
            wid1,
        ),
        (
            Morph("čušup^ha"),
            Morph("čušp^ha"),
            wid2,
        ),
        (
            Morph("išima"),
            Morph("išma"),
            wid3,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "house", "poss": "my"},
        wid2: {"root": "corner", "poss": "his"},
        wid3: {"root": "bed", "poss": "your"},
    }
    segmentation = {
        wid1: [0, 2],
        wid2: [0, 3],
        wid3: [0, 2],
    }
    order = ["root", "poss"]
    morphemes = {
        ("root", "house"): "uta",
        ("poss", "my"): "ŋa",
        ("root", "corner"): "čušu",
        ("poss", "his"): "p^ha",
        ("root", "bed"): "iši",
        ("poss", "your"): "ma",
    }
    candidates = RuleSolver(
        examples=examples,
        bank=FeatureBank(["utaŋčšp^him"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._generate_guard_candidates(
        float("inf"),
        FeatureMatrix([(True, "vowel")]),
        EmptySpecification(),
        freq_cap=15,
    )
    assert normalize_candidates(candidates) == [
        ("", ""),
        ("", "[ -tense ]"),
        ("", "[ -high ]"),
        ("", "[ -back ]"),
        ("", "[ -rounded ]"),
        ("", "[ -vowel ]"),
        ("", "[ -alveolar ]"),
        ("", "[ -low ]"),
        ("", "[ -central ]"),
        ("", "[ -alveopalatal ]"),
        ("", "[ -affricate ]"),
        ("", "[ -fricative ]"),
        ("", "[ -front ]"),
        ("", "+"),
        ("", "+ [ -tense ]"),
    ]


def test_substitution_1():
    wid1 = ("nd^zap", (("root", "sky"), ("poss", "poss")))
    wid2 = ("nd^zay", (("root", "vine"), ("poss", "poss")))
    wid3 = ("ndʌt^s", (("root", "tooth"), ("poss", "poss")))
    examples = [
        (
            Morph("nt^sap"),
            Morph("nd^zap"),
            wid1,
        ),
        (
            Morph("nt^say"),
            Morph("nd^zay"),
            wid2,
        ),
        (
            Morph("ntʌt^s"),
            Morph("ndʌt^s"),
            wid3,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "sky", "poss": "poss"},
        wid2: {"root": "vine", "poss": "poss"},
        wid3: {"root": "tooth", "poss": "poss"},
    }
    segmentation = {
        wid1: [0, 1],
        wid2: [0, 1],
        wid3: [0, 1],
    }
    order = ["poss", "root"]
    morphemes = {
        ("poss", "poss"): "",
        ("root", "sky"): "t^sap",
        ("root", "vine"): "t^say",
        ("root", "tooth"): "tʌt^s",
    }
    candidates = RuleSolver(
        examples=examples,
        bank=FeatureBank(["nt^sd^zapytdʌ"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._generate_guard_candidates(
        float("inf"),
        FeatureMatrix([(False, "vowel")]),
        FeatureMatrix([(True, "voice")]),
        freq_cap=12,
    )
    print(normalize_candidates(candidates))
    assert normalize_candidates(candidates) == [
        ("", ""),
        ("", "[ -alveolar ]"),
        ("", "[ -nasal ]"),
        ("", "[ +voice ]"),
        ("", "[ +sonorant ]"),
        ("", "[ -affricate ]"),
        ("", "[ +central ]"),
        ("", "[ +vowel ]"),
        ("", "[ -bilabial ]"),
        ("", "[ -stop ]"),
        ("", "[ -glide ]"),
        ("+", ""),
    ]


def test_substitution_2():
    wid1 = ("luvaroos", (("root", "vine"), ("big", "big"), ("number", "du")))
    wid2 = ("luxonaleŋ", (("root", "day"), ("number", "du"), ("collective", "part")))
    examples = [
        (
            Morph("lupatoos"),
            Morph("luvaroos"),
            wid1,
        ),
        (
            Morph("lukonaleŋ"),
            Morph("luxonaleŋ"),
            wid2,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "vine", "big": "big", "number": "du"},
        wid2: {"root": "day", "number": "du", "collective": "part"},
    }
    segmentation = {
        wid1: [0, 2, 5],
        wid2: [0, 2, 5],
    }
    order = ["number", "big", "collective", "root"]
    morphemes = {
        ("number", "du"): "lu",
        ("big", "big"): "pat",
        ("collective", "part"): "kon",
        ("root", "vine"): "oos",
        ("root", "day"): "aleŋ",
    }
    solver = RuleSolver(
        examples=examples,
        bank=FeatureBank(["lupvatrosŋkxne"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )
    candidates_p = solver._generate_guard_candidates(
        float("inf"), ConstantPhoneme("p"), ConstantPhoneme("v"), freq_cap=10
    )
    candidates_t = solver._generate_guard_candidates(
        float("inf"), ConstantPhoneme("t"), ConstantPhoneme("r"), freq_cap=10
    )
    candidates_k = solver._generate_guard_candidates(
        float("inf"), ConstantPhoneme("k"), ConstantPhoneme("x"), freq_cap=10
    )
    assert normalize_candidates(candidates_p) == [
        ("", ""),
        ("", "a"),
        ("", "[ -liquid ]"),
        ("", "[ -lateral ]"),
        ("", "[ +voice ]"),
        ("", "[ +sonorant ]"),
        ("", "[ +tense ]"),
        ("", "[ -high ]"),
        ("", "[ -back ]"),
        ("", "[ -rounded ]"),
    ]
    assert normalize_candidates(candidates_t) == [
        ("", ""),
        ("", "o"),
        ("", "[ -liquid ]"),
        ("", "[ -lateral ]"),
        ("", "[ +voice ]"),
        ("", "[ +sonorant ]"),
        ("", "[ +tense ]"),
        ("", "[ -high ]"),
        ("", "[ +back ]"),
        ("", "[ +rounded ]"),
    ]
    assert normalize_candidates(candidates_k) == [
        ("", ""),
        ("", "o"),
        ("", "[ -liquid ]"),
        ("", "[ -lateral ]"),
        ("", "[ +voice ]"),
        ("", "[ +sonorant ]"),
        ("", "[ +tense ]"),
        ("", "[ -high ]"),
        ("", "[ +back ]"),
        ("", "[ +rounded ]"),
    ]


def test_integration():
    wid1 = ("hül", (("root", "turn"),))
    examples = [
        (
            Morph("huli"),
            Morph("hül"),
            wid1,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "turn"},
    }
    segmentation = {
        wid1: [0],
    }
    order = ["root"]
    morphemes = {
        ("root", "turn"): "huli",
    }
    solver = RuleSolver(
        examples=examples,
        bank=FeatureBank(["huüli"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )
    candidates_insertion = solver._generate_guard_candidates(
        float("inf"), EmptySpecification(), ConstantPhoneme("i"), freq_cap=10
    )
    candidates_deletion = solver._generate_guard_candidates(
        float("inf"), ConstantPhoneme("i"), EmptySpecification(), freq_cap=10
    )
    candidates_substitution = solver._generate_guard_candidates(
        float("inf"), ConstantPhoneme("u"), ConstantPhoneme("ü"), freq_cap=10
    )
    restricted_candidates = solver._generate_guard_candidates(
        ConstantPhoneme("i").cost() + 2.0,
        ConstantPhoneme("i"),
        EmptySpecification(),
        freq_cap=10,
    )
    candidates_wrong_focus = solver._generate_guard_candidates(
        float("inf"), ConstantPhoneme("u"), EmptySpecification(), freq_cap=10
    )
    assert normalize_candidates(candidates_insertion) == []
    assert normalize_candidates(candidates_deletion) == [
        ("", ""),
        ("l", ""),
        ("[ -laryngeal ]", ""),
        ("[ -fricative ]", ""),
        ("[ +sonorant ]", ""),
        ("[ +voice ]", ""),
        ("[ -tense ]", ""),
        ("[ -high ]", ""),
        ("[ -back ]", ""),
        ("[ -rounded ]", ""),
    ]
    assert normalize_candidates(candidates_substitution) == [
        ("", ""),
        ("", "l"),
        ("", "[ -laryngeal ]"),
        ("", "[ -fricative ]"),
        ("", "[ +sonorant ]"),
        ("", "[ +voice ]"),
        ("", "[ -tense ]"),
        ("", "[ -high ]"),
        ("", "[ -back ]"),
        ("", "[ -rounded ]"),
    ]
    assert normalize_candidates(restricted_candidates) == [
        ("", ""),
        ("[ -laryngeal ]", ""),
        ("[ -fricative ]", ""),
        ("[ +sonorant ]", ""),
        ("[ +voice ]", ""),
        ("[ -tense ]", ""),
        ("[ -high ]", ""),
        ("[ -back ]", ""),
        ("[ -rounded ]", ""),
        ("[ -vowel ]", ""),
    ]
    assert normalize_candidates(candidates_wrong_focus) == []


def test_metathesis():
    wid1 = (
        "pokskukyʌsmʌ",
        (
            ("root", "chair"),
            ("case", "above"),
        ),
    )
    examples = [
        (
            Morph("pokskuykʌsmʌ"),
            Morph("pokskukyʌsmʌ"),
            wid1,
        ),
    ]
    word_to_slot_values = {wid1: {"root": "chair", "case": "above"}}
    segmentation = {wid1: [0, 7, 12]}
    order = ["root", "case"]
    morphemes = {
        ("root", "chair"): "pokskuy",
        ("case", "above"): "kʌsmʌ",
    }
    candidates = RuleSolver(
        examples=examples,
        bank=FeatureBank(["poksuyʌm"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._generate_guard_candidates(
        float("inf"),
        MetathesisFocus(ConstantPhoneme("y"), ConstantPhoneme("k")),
        MetathesisSpecification(),
        freq_cap=10,
    )
    assert normalize_candidates(candidates) == [
        ("", ""),
        ("", "ʌ"),
        ("", "[ -bilabial ]"),
        ("", "[ -stop ]"),
        ("", "[ +voice ]"),
        ("", "[ +middle ]"),
        ("", "[ -tense ]"),
        ("", "[ -back ]"),
        ("", "[ -rounded ]"),
        ("", "[ +vowel ]"),
    ]


if __name__ == "__main__":
    test_insertion_1()
    test_insertion_2()
    test_deletion_1()
    test_deletion_2()
    test_substitution_1()
    test_substitution_2()
    test_integration()
    test_metathesis()
