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
from rule_search import invert_rule


# rule:    Ø -> n / _ + [ -nasal ]
# inverse: n -> Ø / _ + [ -nasal ]
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

    inv_focus = ConstantPhoneme("n")
    inv_change = EmptySpecification()
    inv_left = Guard("L", False, False, False, [])
    inv_right = Guard(
        "R",
        False,
        False,
        False,
        [BoundarySpecification(), FeatureMatrix([(False, "nasal")])],
    )
    inv_rule = Rule(inv_focus, inv_change, inv_left, inv_right)
    assert invert_rule(rule) == inv_rule


# rule:    Ø -> -1 / [ -vowel ] _ +
# inverse: -1 -> Ø / [ -vowel ] _ +
def test_insertion_2():
    focus = EmptySpecification()
    change = OffsetSpecification(-1)
    left = Guard("L", False, False, False, [FeatureMatrix([(False, "vowel")])])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "neighbor")

    inv_focus = OffsetSpecification(-1)
    inv_change = EmptySpecification()
    inv_left = Guard("L", False, False, False, [FeatureMatrix([(False, "vowel")])])
    inv_right = Guard("R", False, False, False, [BoundarySpecification()])
    inv_rule = Rule(inv_focus, inv_change, inv_left, inv_right)
    assert invert_rule(rule) == inv_rule


# rule:    a -> Ø / _ +
# inverse: Ø -> a / _ +
def test_deletion():
    focus = ConstantPhoneme("a")
    change = EmptySpecification()
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")

    inv_focus = EmptySpecification()
    inv_change = ConstantPhoneme("a")
    inv_left = Guard("L", False, False, False, [])
    inv_right = Guard("R", False, False, False, [BoundarySpecification()])
    inv_rule = Rule(inv_focus, inv_change, inv_left, inv_right)
    assert invert_rule(rule) == inv_rule


# rule:    [ -nasal ] -> [ +voice ] / [ +nasal ] _
# inverse: [ -nasal ] -> [ -voice ] / [ +nasal ] _
def test_substitution_1():
    focus = FeatureMatrix([(False, "nasal")])
    change = FeatureMatrix([(True, "voice")])
    left = Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])])
    right = Guard("R", False, False, False, [])
    rule = Rule(focus, change, left, right)

    inv_focus = FeatureMatrix([(False, "nasal")])
    inv_change = FeatureMatrix([(False, "voice")])
    inv_left = Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])])
    inv_right = Guard("R", False, False, False, [])
    inv_rule = Rule(inv_focus, inv_change, inv_left, inv_right)
    assert invert_rule(rule) == inv_rule


# rule:    [ -voice ] -> [ +voice ] / [ +nasal ] _
# inverse: [ +voice ] -> [ -voice ] / [ +nasal ] _
def test_substitution_2():
    focus = FeatureMatrix([(False, "voice")])
    change = FeatureMatrix([(True, "voice")])
    left = Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])])
    right = Guard("R", False, False, False, [])
    rule = Rule(focus, change, left, right)

    inv_focus = FeatureMatrix([(True, "voice")])
    inv_change = FeatureMatrix([(False, "voice")])
    inv_left = Guard("L", False, False, False, [FeatureMatrix([(True, "nasal")])])
    inv_right = Guard("R", False, False, False, [])
    inv_rule = Rule(inv_focus, inv_change, inv_left, inv_right)
    assert invert_rule(rule) == inv_rule


# rule:    p -> v / + _
# inverse: v -> p / + _
def test_substitution_3():
    focus = ConstantPhoneme("p")
    change = ConstantPhoneme("v")
    left = Guard("L", False, False, False, [BoundarySpecification()])
    right = Guard("R", False, False, False, [])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")

    inv_focus = ConstantPhoneme("v")
    inv_change = ConstantPhoneme("p")
    inv_left = Guard("L", False, False, False, [BoundarySpecification()])
    inv_right = Guard("R", False, False, False, [])
    inv_rule = Rule(inv_focus, inv_change, inv_left, inv_right)
    assert invert_rule(rule) == inv_rule


# rule:    n -> -1 / _ +
# inverse: -1 -> n / _ +
def test_substitution_4():
    focus = ConstantPhoneme("n")
    change = OffsetSpecification(-1)
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")

    inv_focus = OffsetSpecification(-1)
    inv_change = ConstantPhoneme("n")
    inv_left = Guard("L", False, False, False, [])
    inv_right = Guard("R", False, False, False, [BoundarySpecification()])
    inv_rule = Rule(inv_focus, inv_change, inv_left, inv_right)
    assert invert_rule(rule) == inv_rule


# rule:    y k -> MET / _ +
# inverse: k y -> MET / _ +
def test_metathesis():
    focus = MetathesisFocus(ConstantPhoneme("y"), ConstantPhoneme("k"))
    change = MetathesisSpecification()
    left = Guard("L", False, False, False, [])
    right = Guard("R", False, False, False, [BoundarySpecification()])
    rule = Rule(focus, change, left, right)
    setattr(rule, "locality_mode", "allowed_morpheme")

    inv_focus = MetathesisFocus(ConstantPhoneme("k"), ConstantPhoneme("y"))
    inv_rule = Rule(inv_focus, change, left, right)
    assert invert_rule(rule) == inv_rule


if __name__ == "__main__":
    test_insertion_1()
    test_insertion_2()
    test_deletion()
    test_substitution_1()
    test_substitution_2()
    test_substitution_3()
    test_substitution_4()
    test_metathesis()
