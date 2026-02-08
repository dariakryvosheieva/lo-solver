import sys
from pathlib import Path

_bpl_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bpl_dir))

from rule import (
    Rule,
    MetathesisFocus,
    MetathesisSpecification,
    EmptySpecification,
    BoundarySpecification,
    OffsetSpecification,
    ConstantPhoneme,
    FeatureMatrix,
    Guard,
)
from rule_search import can_invert_rule


# Ø -> n / _ + [ -nasal ]
def test_insertion_1():
    focus = EmptySpecification()
    change = ConstantPhoneme("n")
    left = Guard("L", False, False, False, [])
    right = Guard(
        "R",
        False,
        False,
        False,
        [BoundarySpecification(), FeatureMatrix([(False, "nasal")])],
    )
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is True


# Ø -> -1 / [ -vowel ] _ +
def test_insertion_2():
    focus = EmptySpecification()
    change = OffsetSpecification(-1)
    left = Guard("L", False, False, False, [FeatureMatrix([(False, "vowel")])])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "neighbor")
    assert can_invert_rule(rule) is True


# [ +vowel ] -> Ø / _ +
def test_deletion_1():
    focus = FeatureMatrix([(True, "vowel")])
    change = EmptySpecification()
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is False


# a -> Ø / _ +
def test_deletion_2():
    focus = ConstantPhoneme("a")
    change = EmptySpecification()
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is True


# [ -voice ] -> [ +voice ] / [ +nasal ] _
def test_substitution_1():
    focus = FeatureMatrix([(False, "voice")])
    change = FeatureMatrix([(True, "voice")])
    left = Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])])
    right = Guard("R", False, False, False, [])
    rule = Rule(focus, change, left, right)
    assert can_invert_rule(rule) is True


# [ +bilabial ] -> v / + _
def test_substitution_2():
    focus = FeatureMatrix([(True, "bilabial")])
    change = ConstantPhoneme("v")
    left = Guard("L", False, False, False, [BoundarySpecification()])
    right = Guard("R", False, False, False, [])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is False


# p -> v / + _
def test_substitution_3():
    focus = ConstantPhoneme("p")
    change = ConstantPhoneme("v")
    left = Guard("L", False, False, False, [BoundarySpecification()])
    right = Guard("R", False, False, False, [])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is True


# [ -tense ] -> -1 / _ +
def test_substitution_4():
    focus = FeatureMatrix([(False, "tense")])
    change = OffsetSpecification(-1)
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is False


# n -> -1 / _ +
def test_substitution_5():
    focus = ConstantPhoneme("n")
    change = OffsetSpecification(-1)
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is True


# y k -> MET / _ +
def test_metathesis():
    focus = MetathesisFocus(ConstantPhoneme("y"), ConstantPhoneme("k"))
    change = MetathesisSpecification()
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")
    assert can_invert_rule(rule) is True


if __name__ == "__main__":
    test_insertion_1()
    test_insertion_2()
    test_deletion_1()
    test_deletion_2()
    test_substitution_1()
    test_substitution_2()
    test_substitution_3()
    test_substitution_4()
    test_substitution_5()
    test_metathesis()
