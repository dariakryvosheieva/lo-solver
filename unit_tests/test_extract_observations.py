import sys
from pathlib import Path

_bpl_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bpl_dir))

from rule_search import RuleSolver
from features import FeatureBank
from morph import Morph


def normalize_observations(observations):
    return sorted(
        [(obs.id, obs.context, obs.morpheme_boundary_data) for obs in observations]
    )


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
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["ŋkomnt^sapytʌ"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("", "ŋ", "", "k"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"k"}},
            },
        ),
        (
            1,
            ("", "n", "", "t^s"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"t^s"}},
            },
        ),
        (
            2,
            ("", "n", "", "t^s"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"t^s"}},
            },
        ),
        (
            3,
            ("", "n", "", "t"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"t"}},
            },
        ),
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
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["pɨzanetləjuɕvrkis"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("", "n", "a", "n"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            1,
            ("", "n", "n", "e"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"e"}},
            },
        ),
        (
            2,
            ("", "ɕ", "u", "ɕ"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            3,
            ("", "ɕ", "ɕ", "e"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"e"}},
            },
        ),
        (
            4,
            ("", "r", "ə", "r"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            5,
            ("", "r", "r", "e"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"e"}},
            },
        ),
        (
            6,
            ("", "n", "i", "n"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            7,
            ("", "n", "n", "e"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"e"}},
            },
        ),
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
    morphemes = {
        ("poss", "cat"): "kusa",
        ("root", "tongue"): "nama",
    }
    segmentation = {
        wid1: [0, 3],
    }
    order = ["poss", "root"]
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["kusnam"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("a", "", "s", "n"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"n"}},
            },
        ),
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
    morphemes = {
        ("root", "house"): "uta",
        ("poss", "my"): "ŋa",
        ("root", "corner"): "čušu",
        ("poss", "his"): "p^ha",
        ("root", "bed"): "iši",
        ("poss", "your"): "ma",
    }
    segmentation = {
        wid1: [0, 2],
        wid2: [0, 3],
        wid3: [0, 2],
    }
    order = ["root", "poss"]
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["utaŋčšp^him"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("a", "", "t", "ŋ"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"ŋ"}},
            },
        ),
        (
            1,
            ("u", "", "š", "p^h"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"p^h"}},
            },
        ),
        (
            2,
            ("i", "", "š", "m"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"m"}},
            },
        ),
    ]


def test_deletion_3():
    wid1 = (
        "tɨezɨtkət",
        (
            ("root", "lake"),
            ("case", "with"),
            ("number", "pl"),
            ("poss", "your-sg"),
        ),
    )
    wid2 = (
        "cerkuezliɕ",
        (
            ("root", "house"),
            ("case", "from"),
            ("number", "pl"),
        ),
    )
    examples = [
        (
            Morph("tɨnezɨtkət"),
            Morph("tɨezɨtkət"),
            wid1,
        ),
        (
            Morph("cerkunezliɕ"),
            Morph("cerkuezliɕ"),
            wid2,
        ),
    ]
    word_to_slot_values = {
        wid1: {"root": "lake", "case": "with", "number": "pl", "poss": "your-sg"},
        wid2: {"root": "house", "case": "from", "number": "pl"},
    }
    morphemes = {
        ("root", "lake"): "tɨ",
        ("number", "pl"): "nez",
        ("poss", "your-sg"): "ɨt",
        ("case", "with"): "kət",
        ("root", "house"): "cerku",
        ("case", "from"): "liɕ",
    }
    segmentation = {
        wid1: [0, 2, 4, 6],
        wid2: [0, 5, 7, 7],
    }
    order = ["root", "number", "poss", "case"]
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["tɨnezkəcruliɕ"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("n", "", "ɨ", "e"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"ɨ"}},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            1,
            ("n", "", "u", "e"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"u"}},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
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
        ("poss", "poss"): "n",
        ("root", "sky"): "t^sap",
        ("root", "vine"): "t^say",
        ("root", "tooth"): "tʌt^s",
    }
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["nt^sd^zapytdʌ"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("t^s", "d^z", "n", "a"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"n"}},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            1,
            ("t^s", "d^z", "n", "a"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"n"}},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            2,
            ("t", "d", "n", "ʌ"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"n"}},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
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
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["lupvatrosŋkxne"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("p", "v", "u", "a"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"u"}},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            1,
            ("t", "r", "a", "o"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": True, "boundary_plus_spec": {"o"}},
            },
        ),
        (
            2,
            ("k", "x", "u", "o"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"u"}},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
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
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["huüli"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    assert normalize_observations(observations) == [
        (
            0,
            ("u", "ü", "h", "l"),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
        (
            1,
            ("i", "", "l", ""),
            {
                "L": {"boundary_only": False, "spec_plus_boundary": set()},
                "R": {"boundary_only": False, "boundary_plus_spec": set()},
            },
        ),
    ]


def test_metathesis():
    wid1 = ("pokskukyʌsmʌ", (("root", "chair"), ("case", "above"),))
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
    observations = RuleSolver(
        examples=examples,
        bank=FeatureBank(["poksuyʌm"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        morphemes=morphemes,
    )._extract_observations()
    observations_local = RuleSolver(
        examples=examples,
        bank=FeatureBank(["poksuyʌm"]),
        word_to_slot_values=word_to_slot_values,
        segmentation=segmentation,
        order=order,
        allowed_morpheme=("root", "chair"),
        morphemes=morphemes,
    )._extract_observations()
    expected = [
        (
            0,
            (("y", "k"), ("k", "y"), "u", "ʌ"),
            {
                "L": {"boundary_only": True, "spec_plus_boundary": {"u"}},
                "R": {"boundary_only": True, "boundary_plus_spec": {"ʌ"}},
            },
        )
    ]
    assert normalize_observations(observations) == expected
    assert normalize_observations(observations_local) == expected


if __name__ == "__main__":
    test_insertion_1()
    test_insertion_2()
    test_deletion_1()
    test_deletion_2()
    test_deletion_3()
    test_substitution_1()
    test_substitution_2()
    test_integration()
    test_metathesis()
