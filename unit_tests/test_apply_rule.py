import sys
from pathlib import Path

_bpl_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bpl_dir))

from morph import Morph
from features import FeatureBank
from rule import (
    Rule,
    ConstantPhoneme,
    FeatureMatrix,
    Guard,
    MetathesisFocus,
    MetathesisSpecification,
    EmptySpecification,
    OffsetSpecification,
    BoundarySpecification
)
from rule_search import can_invert_rule, invert_rule
from rule_application import apply_rule


def test_insertion_1():
    ur1 = "kom"
    ur2 = "t^sap"
    ur3 = "t^say"
    ur4 = "tʌt^s"
    ur5 = "mok"
    ur6 = "nakpat"

    sr1 = "nkom"
    sr2 = "nt^sap"
    sr3 = "nt^say"
    sr4 = "ntʌt^s"
    sr5 = "mok"
    sr6 = "nakpat"

    wid1 = (sr1, (("root", "post"), ("poss", "poss")))
    wid2 = (sr2, (("root", "sky"), ("poss", "poss")))
    wid3 = (sr3, (("root", "vine"), ("poss", "poss")))
    wid4 = (sr4, (("root", "tooth"), ("poss", "poss")))
    wid5 = (sr5, (("root", "corn"), ("poss", "poss")))
    wid6 = (sr6, (("root", "cactus"), ("poss", "poss")))

    word_to_slot_values = {
        wid1: {"root": "post", "poss": "poss"},
        wid2: {"root": "sky", "poss": "poss"},
        wid3: {"root": "vine", "poss": "poss"},
        wid4: {"root": "tooth", "poss": "poss"},
        wid5: {"root": "corn", "poss": "poss"},
        wid6: {"root": "cactus", "poss": "poss"},
    }

    order = ["poss", "root"]

    morphemes = {
        ("poss", "poss"): "",
        ("root", "post"): "kom",
        ("root", "sky"): "t^sap",
        ("root", "vine"): "t^say",
        ("root", "tooth"): "tʌt^s",
        ("root", "corn"): "mok",
        ("root", "cactus"): "nakpat",
    }

    bank = FeatureBank(["ŋkomnt^sapytʌ"])

    allowed_morpheme = ("poss", "poss")

    rule = Rule(
        EmptySpecification(),
        ConstantPhoneme("n"),
        Guard("L", False, False, False, []),
        Guard("R", False, False, False, [
            BoundarySpecification(),
            FeatureMatrix([(False, "nasal")])
        ])
    )
    setattr(rule, "locality_mode", "allowed_morpheme")

    out1 = apply_rule(
        rule,
        Morph(ur1),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid1,
    )
    out2 = apply_rule(
        rule,
        Morph(ur2),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid2,
    )
    out3 = apply_rule(
        rule,
        Morph(ur3),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid3,
    )
    out4 = apply_rule(
        rule,
        Morph(ur4),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid4,
    )
    out5 = apply_rule(
        rule,
        Morph(ur5),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid5,
    )
    out6 = apply_rule(
        rule,
        Morph(ur6),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid6,
    )

    assert "".join(out1.phonemes) == sr1
    assert "".join(out2.phonemes) == sr2
    assert "".join(out3.phonemes) == sr3
    assert "".join(out4.phonemes) == sr4
    assert "".join(out5.phonemes) == sr5
    assert "".join(out6.phonemes) == sr6


def test_insertion_2():
    ur1 = "pɨzanezɨtlən"
    ur2 = "juɕezə"
    ur3 = "vərezlən"
    ur4 = "kəinezɨs"
    ur5 = "tɨezɨtkət"
    ur6 = "cerkuezliɕ"

    sr1 = "pɨzannezɨtlən"
    sr2 = "juɕɕezə"
    sr3 = "vərrezlən"
    sr4 = "kəinnezɨs"
    sr5 = "tɨezɨtkət"
    sr6 = "cerkuezliɕ"

    wid1 = (sr1, (("root", "desk"), ("case", "of"), ("number", "pl"), ("poss", "your-sg")))
    wid2 = (sr2, (("root", "swan"), ("number", "pl"), ("poss", "my")))
    wid3 = (sr3, (("root", "forest"), ("case", "of"), ("number", "pl")))
    wid4 = (sr4, (("root", "wolf"), ("number", "pl"), ("poss", "his")))
    wid5 = (sr5, (("root", "lake"), ("case", "with"), ("number", "pl"), ("poss", "your-sg")))
    wid6 = (sr6, (("root", "house"), ("case", "from"), ("number", "pl")))

    word_to_slot_values = {
        wid1: {"root": "desk", "case": "of", "number": "pl", "poss": "your-sg"},
        wid2: {"root": "swan", "number": "pl", "poss": "my"},
        wid3: {"root": "forest", "case": "of", "number": "pl"},
        wid4: {"root": "wolf", "number": "pl", "poss": "his"},
        wid5: {"root": "lake", "case": "with", "number": "pl", "poss": "your-sg"},
        wid6: {"root": "house", "case": "from", "number": "pl"},
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

    bank = FeatureBank(["pɨzanetləjuɕvrkisc"])

    allowed_morpheme = ("number", "pl")

    rule = Rule(
        EmptySpecification(),
        OffsetSpecification(-1),
        Guard("L", False, False, False, [BoundarySpecification(), FeatureMatrix([(False, "vowel")])]),
        Guard("R", False, False, False, [])
    )
    setattr(rule, "locality_mode", "allowed_morpheme")

    out1 = apply_rule(
        rule,
        Morph(ur1),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid1,
    )
    out2 = apply_rule(
        rule,
        Morph(ur2),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid2,
    )
    out3 = apply_rule(
        rule,
        Morph(ur3),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid3,
    )
    out4 = apply_rule(
        rule,
        Morph(ur4),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid4,
    )
    out5 = apply_rule(
        rule,
        Morph(ur5),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid5,
    )
    out6 = apply_rule(
        rule,
        Morph(ur6),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid6,
    )

    assert "".join(out1.phonemes) == sr1
    assert "".join(out2.phonemes) == sr2
    assert "".join(out3.phonemes) == sr3
    assert "".join(out4.phonemes) == sr4
    assert "".join(out5.phonemes) == sr5
    assert "".join(out6.phonemes) == sr6


def test_insertion_3():
    ur1 = "pɨzanezɨtlən"
    ur2 = "juɕezə"
    ur3 = "vərezlən"
    ur4 = "kəinezɨs"
    ur5 = "tɨezɨtkət"
    ur6 = "cerkuezliɕ"

    sr1 = "pɨzannezɨtlən"
    sr2 = "juɕɕezə"
    sr3 = "vərrezlən"
    sr4 = "kəinnezɨs"
    sr5 = "tɨezɨtkət"
    sr6 = "cerkuezliɕ"

    wid1 = (sr1, (("root", "desk"), ("case", "of"), ("number", "pl"), ("poss", "your-sg")))
    wid2 = (sr2, (("root", "swan"), ("number", "pl"), ("poss", "my")))
    wid3 = (sr3, (("root", "forest"), ("case", "of"), ("number", "pl")))
    wid4 = (sr4, (("root", "wolf"), ("number", "pl"), ("poss", "his")))
    wid5 = (sr5, (("root", "lake"), ("case", "with"), ("number", "pl"), ("poss", "your-sg")))
    wid6 = (sr6, (("root", "house"), ("case", "from"), ("number", "pl")))

    word_to_slot_values = {
        wid1: {"root": "desk", "case": "of", "number": "pl", "poss": "your-sg"},
        wid2: {"root": "swan", "number": "pl", "poss": "my"},
        wid3: {"root": "forest", "case": "of", "number": "pl"},
        wid4: {"root": "wolf", "number": "pl", "poss": "his"},
        wid5: {"root": "lake", "case": "with", "number": "pl", "poss": "your-sg"},
        wid6: {"root": "house", "case": "from", "number": "pl"},
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

    bank = FeatureBank(["pɨzanetləjuɕvrkisc"])

    allowed_morpheme = ("number", "pl")

    rule = Rule(
        EmptySpecification(),
        OffsetSpecification(-1),
        Guard("L", False, False, False, [FeatureMatrix([(False, "vowel")])]),
        Guard("R", False, False, False, [BoundarySpecification()])
    )
    setattr(rule, "locality_mode", "neighbor")

    out1 = apply_rule(
        rule,
        Morph(ur1),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid1,
    )
    out2 = apply_rule(
        rule,
        Morph(ur2),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid2,
    )
    out3 = apply_rule(
        rule,
        Morph(ur3),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid3,
    )
    out4 = apply_rule(
        rule,
        Morph(ur4),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid4,
    )
    out5 = apply_rule(
        rule,
        Morph(ur5),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid5,
    )
    out6 = apply_rule(
        rule,
        Morph(ur6),
        bank=bank,
        allowed_morpheme=allowed_morpheme,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid6,
    )

    assert "".join(out1.phonemes) == sr1
    assert "".join(out2.phonemes) == sr2
    assert "".join(out3.phonemes) == sr3
    assert "".join(out4.phonemes) == sr4
    assert "".join(out5.phonemes) == sr5
    assert "".join(out6.phonemes) == sr6


def test_deletion_1():
    ur1 = "utaŋa"
    ur2 = "čušup^ha"
    ur3 = "išima"

    sr1 = "utŋa"
    sr2 = "čušp^ha"
    sr3 = "išma"

    wid1 = (sr1, (("root", "house"), ("poss", "my")))
    wid2 = (sr2, (("root", "corner"), ("poss", "his")))
    wid3 = (sr3, (("root", "bed"), ("poss", "your")))

    word_to_slot_values = {
        wid1: {"root": "house", "poss": "my"},
        wid2: {"root": "corner", "poss": "his"},
        wid3: {"root": "bed", "poss": "your"},
    }

    order = ["root", "poss"]

    morphemes = {
        ("root", "house"): "uta",
        ("root", "corner"): "čušu",
        ("root", "bed"): "iši",
        ("poss", "my"): "ŋa",
        ("poss", "his"): "p^ha",
        ("poss", "your"): "ma",
    }

    bank = FeatureBank(["utaŋčšp^him"])

    rule = Rule(
        FeatureMatrix([(True, "vowel")]),
        EmptySpecification(),
        Guard("L", False, False, False, []),
        Guard("R", False, False, False, [BoundarySpecification()])
    )
    setattr(rule, "locality_mode", None)

    out1 = apply_rule(
        rule,
        Morph(ur1),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid1,
    )
    out2 = apply_rule(
        rule,
        Morph(ur2),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid2,
    )
    out3 = apply_rule(
        rule,
        Morph(ur3),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid3,
    )

    assert "".join(out1.phonemes) == sr1
    assert "".join(out2.phonemes) == sr2
    assert "".join(out3.phonemes) == sr3


def test_deletion_2():
    ur1 = "nkom"
    ur2 = "nt^sap"
    ur3 = "nt^say"
    ur4 = "ntʌt^s"
    ur5 = "nmok"
    ur6 = "nnakpat"

    sr1 = "nkom"
    sr2 = "nt^sap"
    sr3 = "nt^say"
    sr4 = "ntʌt^s"
    sr5 = "mok"
    sr6 = "nakpat"

    wid1 = (sr1, (("root", "post"), ("poss", "poss")))
    wid2 = (sr2, (("root", "sky"), ("poss", "poss")))
    wid3 = (sr3, (("root", "vine"), ("poss", "poss")))
    wid4 = (sr4, (("root", "tooth"), ("poss", "poss")))
    wid5 = (sr5, (("root", "corn"), ("poss", "poss")))
    wid6 = (sr6, (("root", "cactus"), ("poss", "poss")))

    word_to_slot_values = {
        wid1: {"root": "post", "poss": "poss"},
        wid2: {"root": "sky", "poss": "poss"},
        wid3: {"root": "vine", "poss": "poss"},
        wid4: {"root": "tooth", "poss": "poss"},
        wid5: {"root": "corn", "poss": "poss"},
        wid6: {"root": "cactus", "poss": "poss"},
    }

    order = ["poss", "root"]

    morphemes = {
        ("poss", "poss"): "n",
        ("root", "post"): "kom",
        ("root", "sky"): "t^sap",
        ("root", "vine"): "t^say",
        ("root", "tooth"): "tʌt^s",
        ("root", "corn"): "mok",
        ("root", "cactus"): "nakpat",
    }

    bank = FeatureBank(["ŋkomnt^sapytʌ"])

    rule = Rule(
        FeatureMatrix([(True, "nasal")]),
        EmptySpecification(),
        Guard("L", False, False, False, []),
        Guard("R", False, False, False, [FeatureMatrix([(True, "nasal")])])
    )
    setattr(rule, "locality_mode", None)

    out1 = apply_rule(
        rule,
        Morph(ur1),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid1,
    )
    out2 = apply_rule(
        rule,
        Morph(ur2),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid2,
    )
    out3 = apply_rule(
        rule,
        Morph(ur3),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid3,
    )
    out4 = apply_rule(
        rule,
        Morph(ur4),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid4,
    )
    out5 = apply_rule(
        rule,
        Morph(ur5),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid5,
    )
    out6 = apply_rule(
        rule,
        Morph(ur6),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid6,
    )

    assert "".join(out1.phonemes) == sr1
    assert "".join(out2.phonemes) == sr2
    assert "".join(out3.phonemes) == sr3
    assert "".join(out4.phonemes) == sr4
    assert "".join(out5.phonemes) == sr5
    assert "".join(out6.phonemes) == sr6


def test_substitution_1():
    ur1 = "nkom"
    ur2 = "nt^sap"
    ur3 = "nt^say"
    ur4 = "ntʌt^s"

    sr1 = "ngom"
    sr2 = "nd^zap"
    sr3 = "nd^zay"
    sr4 = "ndʌt^s"

    wid1 = (sr1, (("root", "post"), ("poss", "poss")))
    wid2 = (sr2, (("root", "sky"), ("poss", "poss")))
    wid3 = (sr3, (("root", "vine"), ("poss", "poss")))
    wid4 = (sr4, (("root", "tooth"), ("poss", "poss")))

    word_to_slot_values = {
        wid1: {"root": "post", "poss": "poss"},
        wid2: {"root": "sky", "poss": "poss"},
        wid3: {"root": "vine", "poss": "poss"},
        wid4: {"root": "tooth", "poss": "poss"},
    }

    order = ["poss", "root"]

    morphemes = {
        ("poss", "poss"): "n",
        ("root", "post"): "kom",
        ("root", "sky"): "t^sap",
        ("root", "vine"): "t^say",
        ("root", "tooth"): "tʌt^s",
    }

    bank = FeatureBank(["ŋkgomnt^sd^zapytdʌ"])

    rule_1 = Rule(
        FeatureMatrix([(False, "voice")]),
        FeatureMatrix([(True, "voice")]),
        Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])]),
        Guard("R", False, False, False, [])
    )
    setattr(rule_1, "locality_mode", None)
    rule_2 = Rule(
        FeatureMatrix([(False, "nasal")]),
        FeatureMatrix([(True, "voice")]),
        Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])]),
        Guard("R", False, False, False, [])
    )
    setattr(rule_2, "locality_mode", None)

    out11 = apply_rule(
        rule_1,
        Morph(ur1),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid1,
    )
    out12 = apply_rule(
        rule_1,
        Morph(ur2),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid2,
    )
    out13 = apply_rule(
        rule_1,
        Morph(ur3),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid3,
    )
    out14 = apply_rule(
        rule_1,
        Morph(ur4),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid4,
    )

    out21 = apply_rule(
        rule_2,
        Morph(ur1),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid1,
    )
    out22 = apply_rule(
        rule_2,
        Morph(ur2),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid2,
    )
    out23 = apply_rule(
        rule_2,
        Morph(ur3),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid3,
    )
    out24 = apply_rule(
        rule_2,
        Morph(ur4),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid4,
    )

    assert "".join(out11.phonemes) == sr1
    assert "".join(out12.phonemes) == sr2
    assert "".join(out13.phonemes) == sr3
    assert "".join(out14.phonemes) == sr4

    assert "".join(out21.phonemes) == sr1
    assert "".join(out22.phonemes) == sr2
    assert "".join(out23.phonemes) == sr3
    assert "".join(out24.phonemes) == sr4


def test_substitution_2():
    ur = "lupatoos"
    sr = "luvaroos"
    wid = (sr, (("root", "vine"), ("big", "big"), ("number", "du")))
    word_to_slot_values = {
        wid: {"root": "vine", "big": "big", "number": "du"},
    }
    order = ["number", "big", "collective", "root"]
    morphemes = {
        ("number", "du"): "lu",
        ("big", "big"): "pat",
        ("root", "vine"): "oos",
    }
    bank = FeatureBank(["lupvatrosŋkxne"])
    rule_1 = Rule(
        ConstantPhoneme("p"),
        ConstantPhoneme("v"),
        Guard("L", False, False, False, [BoundarySpecification()]),
        Guard("R", False, False, False, [])
    )
    setattr(rule_1, "locality_mode", None)
    rule_2 = Rule(
        ConstantPhoneme("t"),
        ConstantPhoneme("r"),
        Guard("L", False, False, False, []),
        Guard("R", False, False, False, [BoundarySpecification()])
    )
    setattr(rule_2, "locality_mode", None)
    out1 = apply_rule(
        rule_1,
        Morph(ur),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid,
    )
    out2 = apply_rule(
        rule_2,
        out1,
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid,
    )
    assert "".join(out2.phonemes) == sr


def test_metathesis():
    ur = "pokskuykʌsmʌ"
    sr = "pokskukyʌsmʌ"
    wid = (sr, (("root", "chair"), ("case", "above"),))
    word_to_slot_values = {wid: {"root": "chair", "case": "above"}}
    order = ["root", "case"]
    morphemes = {
        ("root", "chair"): "pokskuy",
        ("case", "above"): "kʌsmʌ",
    }
    bank = FeatureBank(["poksuyʌm"])
    rule = Rule(
        MetathesisFocus(ConstantPhoneme("y"), ConstantPhoneme("k")),
        MetathesisSpecification(),
        Guard("L", False, False, False, [BoundarySpecification()]),
        Guard("R", False, False, False, [BoundarySpecification()])
    )
    setattr(rule, "locality_mode", None)
    out1 = apply_rule(
        rule,
        Morph(ur),
        bank=bank,
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid,
    )
    setattr(rule, "locality_mode", "allowed_morpheme")
    out2 = apply_rule(
        rule,
        Morph(ur),
        bank=bank,
        allowed_morpheme=("root", "chair"),
        morphemes=morphemes,
        order=order,
        word_to_slot_values=word_to_slot_values,
        wid=wid,
    )
    assert "".join(out1.phonemes) == sr
    assert "".join(out2.phonemes) == sr


def test_inverse_1():
    ur1 = "ngom"
    ur2 = "nd^zap"
    ur3 = "nd^zay"
    ur4 = "ntʌt^s"

    sr1 = "ngom"
    sr2 = "nd^zap"
    sr3 = "nd^zay"
    sr4 = "ndʌt^s"

    wid1 = (sr1, (("root", "post"), ("poss", "poss")))
    wid2 = (sr2, (("root", "sky"), ("poss", "poss")))
    wid3 = (sr3, (("root", "vine"), ("poss", "poss")))
    wid4 = (sr4, (("root", "tooth"), ("poss", "poss")))

    word_to_slot_values = {
        wid1: {"root": "post", "poss": "poss"},
        wid2: {"root": "sky", "poss": "poss"},
        wid3: {"root": "vine", "poss": "poss"},
        wid4: {"root": "tooth", "poss": "poss"},
    }

    order = ["poss", "root"]

    morphemes = {
        ("poss", "poss"): "n",
        ("root", "post"): "gom",
        ("root", "sky"): "d^zap",
        ("root", "vine"): "d^zay",
        ("root", "tooth"): "tʌt^s",
    }

    bank = FeatureBank(["ŋkgomnt^sd^zapytdʌ"])

    rule = Rule(
        FeatureMatrix([(False, "voice")]),
        FeatureMatrix([(True, "voice")]),
        Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])]),
        Guard("R", False, False, False, [])
    )
    setattr(rule, "locality_mode", None)

    if can_invert_rule(rule):
        inv_rule = invert_rule(rule)
        setattr(inv_rule, "locality_mode", None)
        out11 = apply_rule(
            inv_rule,
            Morph(ur1),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid1,
        )
        out12 = apply_rule(
            inv_rule,
            Morph(ur2),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid2,
        )
        out13 = apply_rule(
            inv_rule,
            Morph(ur3),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid3,
        )
        out14 = apply_rule(
            inv_rule,
            Morph(ur4),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid4,
        )
        out21 = apply_rule(
            rule,
            Morph(out11),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid1,
        )
        out22 = apply_rule(
            rule,
            Morph(out12),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid2,
        )
        out23 = apply_rule(
            rule,
            Morph(out13),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid3,
        )
        out24 = apply_rule(
            rule,
            Morph(out14),
            bank=bank,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid4,
        )
        assert "".join(out21.phonemes) == sr1
        assert "".join(out22.phonemes) == sr2
        assert "".join(out23.phonemes) == sr3
        assert "".join(out24.phonemes) == sr4

    
def test_inverse_2():
    ur1 = "pɨzannezɨtlən"
    ur2 = "juɕezə"
    ur3 = "vərrezlən"
    ur4 = "kəinnezɨs"
    ur5 = "tɨezɨtkət"
    ur6 = "cerkuezliɕ"

    sr1 = "pɨzannezɨtlən"
    sr2 = "juɕɕezə"
    sr3 = "vərrezlən"
    sr4 = "kəinnezɨs"
    sr5 = "tɨezɨtkət"
    sr6 = "cerkuezliɕ"

    wid1 = (sr1, (("root", "desk"), ("case", "of"), ("number", "pl"), ("poss", "your-sg")))
    wid2 = (sr2, (("root", "swan"), ("number", "pl"), ("poss", "my")))
    wid3 = (sr3, (("root", "forest"), ("case", "of"), ("number", "pl")))
    wid4 = (sr4, (("root", "wolf"), ("number", "pl"), ("poss", "his")))
    wid5 = (sr5, (("root", "lake"), ("case", "with"), ("number", "pl"), ("poss", "your-sg")))
    wid6 = (sr6, (("root", "house"), ("case", "from"), ("number", "pl")))

    word_to_slot_values = {
        wid1: {"root": "desk", "case": "of", "number": "pl", "poss": "your-sg"},
        wid2: {"root": "swan", "number": "pl", "poss": "my"},
        wid3: {"root": "forest", "case": "of", "number": "pl"},
        wid4: {"root": "wolf", "number": "pl", "poss": "his"},
        wid5: {"root": "lake", "case": "with", "number": "pl", "poss": "your-sg"},
        wid6: {"root": "house", "case": "from", "number": "pl"},
    }

    order = ["root", "number", "poss", "case"]

    morphemes = {
        ("root", "desk"): "pɨzann",
        ("root", "swan"): "juɕ",
        ("root", "forest"): "vərr",
        ("root", "wolf"): "kəinn",
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

    bank = FeatureBank(["pɨzanetləjuɕvrkisc"])

    allowed_morpheme = ("number", "pl")

    rule = Rule(
        EmptySpecification(),
        OffsetSpecification(-1),
        Guard("L", False, False, False, [FeatureMatrix([(False, "vowel")])]),
        Guard("R", False, False, False, [BoundarySpecification()])
    )
    setattr(rule, "locality_mode", "neighbor")
    
    if can_invert_rule(rule):
        inv_rule = invert_rule(rule)
        setattr(inv_rule, "locality_mode", "neighbor")

        out11 = apply_rule(
            inv_rule,
            Morph(ur1),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid1,
        )
        out12 = apply_rule(
            inv_rule,
            Morph(ur2),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid2,
        )
        out13 = apply_rule(
            inv_rule,
            Morph(ur3),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid3,
        )
        out14 = apply_rule(
            inv_rule,
            Morph(ur4),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid4,
        )
        out15 = apply_rule(
            inv_rule,
            Morph(ur5),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid5,
        )
        out16 = apply_rule(
            inv_rule,
            Morph(ur6),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid6,
        )

        revised_morphemes = {
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

        out21 = apply_rule(
            rule,
            Morph(out11),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=revised_morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid1,
        )
        out22 = apply_rule(
            rule,
            Morph(out12),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=revised_morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid2,
        )
        out23 = apply_rule(
            rule,
            Morph(out13),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=revised_morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid3,
        )
        out24 = apply_rule(
            rule,
            Morph(out14),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=revised_morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid4,
        )
        out25 = apply_rule(
            rule,
            Morph(out15),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=revised_morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid5,
        )   
        out26 = apply_rule(
            rule,
            Morph(out16),
            bank=bank,
            allowed_morpheme=allowed_morpheme,
            morphemes=revised_morphemes,
            order=order,
            word_to_slot_values=word_to_slot_values,
            wid=wid6,
        )

        assert "".join(out21.phonemes) == sr1
        assert "".join(out22.phonemes) == sr2
        assert "".join(out23.phonemes) == sr3
        assert "".join(out24.phonemes) == sr4
        assert "".join(out25.phonemes) == sr5
        assert "".join(out26.phonemes) == sr6


if __name__ == "__main__":
    test_insertion_1()
    test_insertion_2()
    test_insertion_3()
    test_deletion_1()
    test_deletion_2()
    test_substitution_1()
    test_substitution_2()
    test_metathesis()
    test_inverse_1()
    test_inverse_2()
