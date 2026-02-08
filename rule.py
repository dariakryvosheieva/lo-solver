import itertools
from random import choice, random

from features import *
from utilities import *


class Braces:
    """
    SPE-style brackets for rule alternatives.
    """

    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def __unicode__(self):
        return "{ " + str(self.r1) + " || " + str(self.r2) + " }"

    def __str__(self):
        return self.__unicode__()

    def cost(self):
        k = self.r1.cost() + self.r2.cost()
        # do not incur cost of repeating exactly the same structure
        if isinstance(self.r1, ConstantPhoneme) and isinstance(
            self.r2, ConstantPhoneme
        ):
            k -= 1
        if isinstance(self.r1, FeatureMatrix) and isinstance(self.r2, FeatureMatrix):
            k -= 1
        return k


# abstract class for focus/change
class FC:
    def __init__(self):
        pass


class Specification:
    def __init__(self):
        pass

    def isDegenerate(self):
        """Whether this specification can be equivalently represented using fewer features"""
        assert False, "isDegenerate: not implemented"

    @staticmethod
    def enumeration(b, cost):
        return ConstantPhoneme.enumeration(b, cost) + FeatureMatrix.enumeration(b, cost)


class ConstantPhoneme(Specification, FC):
    def __init__(self, p):
        self.p = p

    def __unicode__(self):
        if self.p == "-":
            return "σ"
        else:
            return self.p

    def __str__(self):
        return self.__unicode__()

    def doesNothing(self):
        return False

    def cost(self):
        """
        Feature-based cost for a constant phoneme specification.
        
        This is the number of features of that phoneme. The separate
        rule-level metric adds an extra 1 per *element* (phoneme, matrix,
        boundary, offset, place), so this cost only reflects feature
        complexity.
        """
        features = featureMap.get(self.p, [])
        return float(len(features))

    def skeleton(self):
        return "K"

    def mutate(self, bank):
        return ConstantPhoneme(choice(bank.phonemes))

    def isDegenerate(self):
        return False

    def violatesGeometry(self):
        return False

    def makeGeometric(self):
        return self

    def share(self, table):
        k = ("CONSTANT", str(self))
        if k in table:
            return table[k]
        table[k] = self
        return self

    def merge(self, other):
        if isinstance(other, ConstantPhoneme) and other.p == self.p:
            return self
        return Braces(self, other)

    def matches(self, test):
        return set(featureMap[self.p]) == set(test)

    def apply(self, test):
        return featureMap[self.p]

    @staticmethod
    def enumeration(b, cost):
        if cost > 1:
            return [ConstantPhoneme(p) for p in b.phonemes]
        return []

    def extension(self, b):
        return [self.p]

    def isDegenerate(self):
        return False
    
    def __eq__(self, other):
        return isinstance(other, ConstantPhoneme) and self.p == other.p


class EmptySpecification(FC):
    def __init__(self):
        pass

    def __unicode__(self):
        return "Ø"

    def __str__(self):
        return self.__unicode__()

    def doesNothing(self):
        return False

    def skeleton(self):
        return "0"

    def cost(self):
        """
        Empty specification (deletion / no change) has zero cost.
        It does not introduce any new meaningful element.
        """
        return 0.0

    def mutate(self, _):
        return self

    def isDegenerate(self):
        return False

    def violatesGeometry(self):
        return True

    def makeGeometric(self):
        return self

    def share(self, table):
        k = ("EMPTYSPECIFICATION", str(self))
        if k in table:
            return table[k]
        table[k] = self
        return self

    def merge(self, other):
        if isinstance(other, EmptySpecification):
            return self
        return Braces(self, other)

    def matches(self, test):
        return True

    def apply(self, test):
        raise Exception("cannot apply deletion rule")

    def __eq__(self, other):
        return isinstance(other, EmptySpecification)


class OffsetSpecification(FC):
    def __init__(self, offset):
        if offset == 0:
            print("WARNING: 0 offset. Not sure if this is a bug or not!")
        # assert offset != 0
        self.offset = offset

    def __unicode__(self):
        return str(self.offset)

    def __str__(self):
        return self.__unicode__()

    def doesNothing(self):
        return False

    def skeleton(self):
        return "Z"

    def cost(self):
        """
        Offset/copy specifications do not introduce features themselves.
        They have zero *feature* cost; the rule-level metric will count
        them as elements separately.
        """
        return 0.0

    def mutate(self, _):
        return self

    def isDegenerate(self):
        return False

    def violatesGeometry(self):
        return False

    def makeGeometric(self):
        return self

    def share(self, table):
        k = ("OFFSETSPECIFICATION", str(self))
        if k in table:
            return table[k]
        table[k] = self
        return self

    def merge(self, other):
        if isinstance(other, OffsetSpecification) and self.offset == other.offset:
            return self
        return Braces(self, other)

    def matches(self, test):
        raise Exception("cannot match offset")

    def apply(self, test):
        raise Exception("cannot apply offset")

    def __eq__(self, other):
        return isinstance(other, OffsetSpecification) and self.offset == other.offset


class MetathesisFocus(FC):
    """
    Focus for a (local) metathesis rule: two adjacent specifications that must
    match a pair of segments in order (s1 s2). Application swaps them to (s2 s1).

    This is intentionally minimal: it is only used by the Python-side
    `RuleSolver` / `apply_rule` path (not by the Sketch parser/enumerator).
    """

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def __unicode__(self):
        # Index-based metathesis focus (value-agnostic): swap the 1st and 2nd
        # phoneme after the left guard.
        if self.s1 is None and self.s2 is None:
            return "1 2"
        return f"{self.s1} {self.s2}"

    def __str__(self):
        return self.__unicode__()

    def doesNothing(self):
        return False

    def skeleton(self):
        return "XY"

    def cost(self):
        # Feature-cost only; element-count is handled in Rule.cost().
        return float(getattr(self.s1, "cost", lambda: 0.0)()) + float(
            getattr(self.s2, "cost", lambda: 0.0)()
        )

    def element_count(self):
        # Two meaningful elements (the two adjacent specifications).
        return 2

    def __eq__(self, other):
        return isinstance(other, MetathesisFocus) and self.s1 == other.s1 and self.s2 == other.s2


class MetathesisSpecification(FC):
    """
    Marker structural change for metathesis. It does not introduce feature cost;
    the "work" is swapping the two focused segments.
    """

    def __init__(self):
        pass

    def __unicode__(self):
        return "MET"

    def __str__(self):
        return self.__unicode__()

    def doesNothing(self):
        return False

    def skeleton(self):
        return "MET"

    def cost(self):
        return 0.0

    def element_count(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, MetathesisSpecification)


class PlaceSpecification(FC):
    def __init__(self, offset):
        # assert offset != 0
        self.offset = offset

    def __unicode__(self):
        return str("place%d" % self.offset)

    def __str__(self):
        return self.__unicode__()

    def doesNothing(self):
        return False

    def skeleton(self):
        return "Z"

    def cost(self):
        """
        Place specifications have zero feature cost; they are counted as
        elements at the rule level.
        """
        return 0.0

    def mutate(self, _):
        return self

    def isDegenerate(self):
        return False

    def violatesGeometry(self):
        return False

    def makeGeometric(self):
        return self

    def share(self, table):
        k = ("PLACESPECIFICATION", str(self))
        if k in table:
            return table[k]
        table[k] = self
        return self

    def merge(self, other):
        if isinstance(other, PlaceSpecification) and self.offset == other.offset:
            return self
        return Braces(self, other)

    def matches(self, test):
        raise Exception("cannot match place")

    def apply(self, test):
        raise Exception("cannot apply place")
    
    def __eq__(self, other):
        return isinstance(other, PlaceSpecification) and self.offset == other.offset


class BoundarySpecification(Specification):
    def __init__(self):
        pass

    def __unicode__(self):
        return "+"

    def __str__(self):
        return self.__unicode__()

    def doesNothing(self):
        return False

    def skeleton(self):
        return "+"

    def cost(self):
        """
        Boundary specifications (+ / #) have zero feature cost; they are
        treated as separate elements at the rule level.
        """
        return 0.0

    def mutate(self, _):
        return self

    def extension(self, _):
        return "+"

    def isDegenerate(self):
        return False

    def violatesGeometry(self):
        return False

    def makeGeometric(self):
        return self

    def share(self, table):
        k = ("BOUNDARYSPECIFICATION", str(self))
        if k in table:
            return table[k]
        table[k] = self
        return self

    def merge(self, other):
        if isinstance(other, BoundarySpecification):
            return self
        return Braces(self, other)

    def matches(self, test):
        return False

    def apply(self, test):
        raise Exception("cannot apply boundary specification")

    def __eq__(self, other):
        return isinstance(other, BoundarySpecification)


class FeatureMatrix(Specification, FC):
    def __init__(self, featuresAndPolarities):
        self.featuresAndPolarities = featuresAndPolarities
        self.representation = None  # string representation

    def mutate(self, bank):
        # delete a feature
        if self.featuresAndPolarities != [] and random() < 0.5:
            toRemove = choice(self.featuresAndPolarities)
            return FeatureMatrix(
                [fp for fp in self.featuresAndPolarities if fp != toRemove]
            )
        else:
            fp = (choice([True, False]), choice(bank.features))
            return FeatureMatrix(list(set(self.featuresAndPolarities + [fp])))

    def violatesGeometry(self):
        fs = {f for _, f in self.featuresAndPolarities}
        if len(fs & VOWELFEATURES) > 0 and "vowel" not in fs:
            return True
        negativeFeatures = {f for p, f in self.featuresAndPolarities if not p}
        if len(negativeFeatures & (CONSONANTPLACEFEATURES | CONSONANTFEATURES)) > 0:
            # this would appear to be excluding vowels!
            # we better make sure that it actually excludes them :)
            extension = self.extension(FeatureBank.GLOBAL)
            if any("vowel" in FeatureBank.GLOBAL.featureMap[p] for p in extension):
                return True
        return False

    def __lt__(self, o):
        assert isinstance(o, FeatureMatrix)

        if len(self.featuresAndPolarities) >= len(o.featuresAndPolarities):
            return False
        for fp in self.featuresAndPolarities:
            if fp not in o.featuresAndPolarities:
                return False
        return True

    def makeGeometric(self, bank=None):
        original = self
        if any(
            (True, vf) in self.featuresAndPolarities for vf in VOWELFEATURES
        ) and "vowel" not in [f for _, f in self.featuresAndPolarities]:
            self = FeatureMatrix(
                list({(True, "vowel")} | set(self.featuresAndPolarities))
            )
        # remove redundant features
        if (True, "vowel") in self.featuresAndPolarities:
            self = FeatureMatrix(
                [
                    (p, f)
                    for p, f in self.featuresAndPolarities
                    if f not in DEFAULTVOWELFEATURES
                ]
            )

        if (
            any(
                (True, vf) in self.featuresAndPolarities
                for vf in CONSONANTPLACEFEATURES
            )
            and sonorant not in {f for _, f in self.featuresAndPolarities}
            and "vowel" not in {f for _, f in self.featuresAndPolarities}
            and len(self.featuresAndPolarities) < 3
        ):
            self = FeatureMatrix(
                list({(False, "vowel")} | set(self.featuresAndPolarities))
            )

        if (False, "vowel") in self.featuresAndPolarities and (
            True,
            "vowel",
        ) in self.featuresAndPolarities:
            print(
                "Got an impossible feature matrix namely",
                self,
                "starting from",
                original,
            )
            assert False

        return self

    def isDegenerate(self):
        if self.doesNothing:
            return False
        e = frozenset(self.extension(FeatureBank.GLOBAL))
        # print "checking if",self,"is degenerate",e
        simplifications = [
            FeatureMatrix([fpp for fpp in self.featuresAndPolarities if fpp != fp])
            for fp in self.featuresAndPolarities
        ]
        # print "simplifications"
        # for s in simplifications:
        #     print s, frozenset(s.extension(FeatureBank.GLOBAL))
        # print
        return any(
            e == frozenset(s.extension(FeatureBank.GLOBAL)) for s in simplifications
        )

    @staticmethod
    def strPolarity(p):
        return "+" if p == True else ("-" if p == False else p)

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        if not hasattr(self, "representation") or self.representation == None:
            elements = sorted(
                [
                    FeatureMatrix.strPolarity(polarity) + f
                    for polarity, f in self.featuresAndPolarities
                ]
            )
            self.representation = "[ {} ]".format(" ".join(elements))
        return self.representation

    def doesNothing(self):
        return len(self.featuresAndPolarities) == 0

    def cost(self):
        """
        Feature-based cost for a feature matrix.
        
        This is the number of specified features. The rule-level metric
        adds a separate 1 per matrix as a *meaningful element*; this
        value only tracks feature complexity.
        """
        return float(len(self.featuresAndPolarities))

    def skeleton(self):
        if self.featuresAndPolarities == []:
            return "[ ]"
        else:
            return "[ +/-F ]"

    def share(self, table):
        k = ("MATRIX", str(self))
        if k in table:
            return table[k]
        table[k] = self
        return self

    def merge(self, other):
        if isinstance(other, FeatureMatrix):
            if {f for _, f in self.featuresAndPolarities} == {
                f for _, f in other.featuresAndPolarities
            }:
                # introduce feature variables
                y = {f: p for p, f in self.featuresAndPolarities}
                x = {f: p for p, f in other.featuresAndPolarities}
                fs = [
                    (
                        (
                            x[f]
                            if x[f] == y[f]
                            else FeatureMatrix.strPolarity(x[f])
                            + "/"
                            + FeatureMatrix.strPolarity(y[f])
                        ),
                        f,
                    )
                    for f in y
                ]
                return FeatureMatrix(fs)

        return Braces(self, other)

    def matches(self, test):
        for p, f in self.featuresAndPolarities:
            if p:
                if not (f in test):
                    return False
            else:
                if f in test:
                    return False
        return True

    def extension(self, bank):
        return [p for p in bank.phonemes if self.matches(featureMap[p])]

    def apply(self, test):
        for p, f in self.featuresAndPolarities:
            if p:
                test = test + [f]
                # mutually exclusive features
                for k in FeatureBank.mutuallyExclusiveClasses:
                    if f in k:
                        test = [_f for _f in test if ((not _f in k) or _f == f)]
                        # Assumption: exclusive classes are themselves mutually exclusive
                        break
            else:
                test = [_f for _f in test if not _f == f]
        return list(set(test))

    @staticmethod
    def enumeration(b, cost):
        if False:
            if cost < 1:
                return []
            cost -= 1
            results = []
            for k in range(cost + 1):
                for features in itertools.combinations(b.features, k):
                    for polarities in itertools.product(*([(True, False)] * k)):
                        results.append(FeatureMatrix(list(zip(polarities, features))))
            return results
        else:
            if cost < 1:
                return []
            cost -= 1
            results = {}
            for k in range(cost + 1):
                for features in itertools.combinations(b.features, k):
                    for polarities in itertools.product(*([(True, False)] * k)):
                        matrix = FeatureMatrix(list(zip(polarities, features)))
                        extension = frozenset(matrix.extension(b))
                        if not (extension in results):
                            results[extension] = matrix
            return list(results.values())

    def __eq__(self, other):
        return isinstance(other, FeatureMatrix) and self.featuresAndPolarities == other.featuresAndPolarities


class Guard:
    def __init__(self, side, endOfString, optionalEnding, starred, specifications):
        self.side = side
        self.endOfString = endOfString
        self.optionalEnding = optionalEnding
        assert not (optionalEnding and endOfString)
        self.starred = starred
        self.specifications = [s for s in specifications if s != None]
        assert len(self.specifications) <= 2
        self.representation = None  # Unicode representation

    def violatesGeometry(self):
        return any(s.violatesGeometry for s in self.specifications)

    def makeGeometric(self):
        return Guard(
            self.side,
            self.endOfString,
            self.optionalEnding,
            self.starred,
            [s.makeGeometric() for s in self.specifications],
        )

    def isDegenerate(self):
        return any(s.isDegenerate() for s in self.specifications)

    def doesNothing(self):
        return not self.endOfString and len(self.specifications) == 0

    def cost(self):
        """
        Complexity cost for a guard.
        
        A guard's cost is the sum of the costs of its specifications.
        Each non-empty specification (phoneme, feature matrix, boundary,
        offset, place) is one meaningful element; booleans (starred,
        endOfString, optionalEnding) are free.
        """
        return float(sum(s.cost() for s in self.specifications))

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        if not hasattr(self, "representation") or self.representation == None:
            parts = []
            parts += list(map(str, self.specifications))
            if self.starred:
                parts[-2] += "*"
            if self.endOfString:
                parts += ["#"]
            if self.optionalEnding:
                parts[-1] = "{#,%s}" % parts[-1]
            if self.side == "L":
                parts.reverse()
            self.representation = " ".join(parts)
        return self.representation

    def pretty(self, copyOffset):
        parts = []
        parts += list(map(str, self.specifications))
        if self.starred:
            parts[-2] += "*"
        if self.optionalEnding:
            parts[-1] = "{#,%s}" % parts[-1]
        if copyOffset != 0:
            if copyOffset < 0 and self.side == "L":
                idx = -copyOffset - 1
                if 0 <= idx < len(parts):
                    parts[idx] += "ᵢ"
            if copyOffset > 0 and self.side == "R":
                idx = copyOffset - 1
                if 0 <= idx < len(parts):
                    parts[idx] += "ᵢ"
        if self.endOfString:
            parts += ["#"]
        if self.side == "L":
            parts.reverse()
        return " ".join(parts)

    def skeleton(self):
        parts = []
        parts += [spec.skeleton() for spec in self.specifications]
        if self.starred:
            parts[-2] += "*"
        if self.endOfString:
            parts += ["#"]
        if self.side == "L":
            parts.reverse()
        return " ".join(parts)

    def share(self, table):
        k = ("GUARD", self.side, str(self))
        if k in table:
            return table[k]
        table[k] = Guard(
            self.side,
            self.endOfString,
            self.optionalEnding,
            self.starred,
            [s.share(table) for s in self.specifications],
        )
        return table[k]

    def merge(self, other):
        assert other.side == self.side
        if (
            self.endOfString != other.endOfString
            or self.starred != other.starred
            or len(self.specifications) != len(other.specifications)
        ):
            return Braces(self, other)
        return Guard(
            self.side,
            self.endOfString,
            self.starred,
            [x.merge(y) for x, y in zip(self.specifications, other.specifications)],
        )

    @staticmethod
    def enumeration(side, b, cost):
        results = []
        for ending in [False, True]:
            for numberOfSpecifications in range(3):
                for starred in (
                    [False] if numberOfSpecifications < 2 else [True, False]
                ):
                    if numberOfSpecifications == 0:
                        if int(starred) + int(ending) <= cost:
                            results.append(Guard(side, ending, False, starred, []))
                    elif numberOfSpecifications == 1:
                        for s in Specification.enumeration(
                            b, int(cost - int(starred) - int(ending))
                        ):
                            results.append(Guard(side, ending, False, starred, [s]))
                    elif numberOfSpecifications == 2:
                        for s1 in Specification.enumeration(
                            b, int(cost - int(starred) - int(ending))
                        ):
                            for s2 in Specification.enumeration(
                                b, int(cost - int(starred) - int(ending) - s1.cost())
                            ):
                                results.append(Guard(side, ending, False, starred, [s1, s2]))
                    else:
                        assert False
        return results

    def __eq__(self, other):
        return isinstance(other, Guard) and self.side == other.side and self.endOfString == other.endOfString and self.optionalEnding == other.optionalEnding and self.starred == other.starred and self.specifications == other.specifications


class Rule:
    def __init__(self, focus, structuralChange, leftTriggers, rightTriggers):
        self.focus = focus
        self.structuralChange = structuralChange
        self.leftTriggers = leftTriggers
        self.rightTriggers = rightTriggers
        self.representation = None  # Unicode representation

    def __lt__(self, o):
        assert isinstance(o, Rule)
        return (
            isinstance(self.structuralChange, FeatureMatrix)
            and isinstance(o.structuralChange, FeatureMatrix)
            and self.structuralChange < o.structuralChange
            and str(self.focus) == str(o.focus)
            and str(self.leftTriggers) == str(o.leftTriggers)
            and str(self.rightTriggers) == str(o.rightTriggers)
        )

    def violatesGeometry(self):
        return (
            self.focus.violatesGeometry()
            or self.leftTriggers.violatesGeometry()
            or self.rightTriggers.violatesGeometry()
        )

    def makeGeometric(self, bank=None):
        focus = self.focus.makeGeometric()
        change = self.structuralChange
        # check to see if change could only ever apply to vowels
        if isinstance(change, FeatureMatrix) and isinstance(focus, FeatureMatrix):
            fs = {f for _, f in change.featuresAndPolarities}
            if (
                len(
                    {f for p, f in change.featuresAndPolarities if p}
                    & (VOWELFEATURES | {"vowel"})
                )
                > 0
                and len(fs) < 3
            ):
                newFocus = {(True, "vowel")} | set(focus.featuresAndPolarities)
                focus = FeatureMatrix(list(newFocus)).makeGeometric()
        # check if the change could only ever apply to consonants
        if isinstance(change, FeatureMatrix) and isinstance(focus, FeatureMatrix):
            fs = {f for p, f in change.featuresAndPolarities if p}
            focus_fs = {f for p, f in focus.featuresAndPolarities}
            if (
                len(fs & CONSONANTFEATURES) > 0
                and len(fs) < 3
                and len(focus_fs & {sonorant}) == 0
            ):
                newFocus = {(False, "vowel")} | set(focus.featuresAndPolarities)
                if {(False, "vowel"), (True, "vowel")} <= newFocus:
                    print(
                        "generated impossible vowel constraints:\t",
                        focus,
                        change,
                        "\tvs:\t",
                        newFocus,
                    )
                    assert False
                focus = FeatureMatrix(list(newFocus)).makeGeometric()

        # Check for redundant structural change
        if isinstance(focus, FeatureMatrix) and isinstance(change, FeatureMatrix):
            # Remove any features which are always true for phonemes matching the focus
            bank = bank or FeatureBank.GLOBAL
            matches = focus.extension(bank)
            change = {
                (polarity, feature)
                for polarity, feature in change.featuresAndPolarities
                if any(
                    (feature in bank.featureMap[p] and not polarity)
                    or (feature not in bank.featureMap[p] and polarity)
                    for p in matches
                )
            }
            change = FeatureMatrix(list(change))
        return Rule(
            focus,
            change,
            self.leftTriggers.makeGeometric(),
            self.rightTriggers.makeGeometric(),
        )

    def isDegenerate(self):
        return (
            self.focus.isDegenerate()
            or self.structuralChange.isDegenerate()
            or self.leftTriggers.isDegenerate()
            or self.rightTriggers.isDegenerate()
        )

    def isGeminiRule(self):
        return (
            isinstance(self.focus, OffsetSpecification)
            and self.focus.offset == 1
            and isinstance(self.structuralChange, EmptySpecification)
        )

    def isCopyRule(self):
        return isinstance(self.structuralChange, OffsetSpecification)

    def merge(self, other):
        return Rule(
            self.focus.merge(other.focus),
            self.structuralChange.merge(other.structuralChange),
            self.leftTriggers.merge(other.leftTriggers),
            self.rightTriggers.merge(other.rightTriggers),
        )

    def share(self, table):
        k = ("RULE", str(self))
        if k in table:
            return table[k]
        table[k] = Rule(
            self.focus.share(table),
            self.structuralChange.share(table),
            self.leftTriggers.share(table),
            self.rightTriggers.share(table),
        )
        return table[k]

    def cost(self):
        """
        Complexity cost for a rule.
        
        Two components:
        - feature cost: sum of the .cost() values of focus, change, and guards
          (ConstantPhoneme / FeatureMatrix = number of features, others 0);
        - element cost: +1 for each meaningful element (phoneme, feature
          matrix, boundary, offset/place, Ø) present in the rule.
        
        There is **no** per-character string cost; elements are counted
        structurally instead.
        """
        if self.doesNothing():
            return 0.0

        # 1. Feature-based cost from specifications
        feature_cost = 0.0
        feature_cost += float(self.focus.cost())
        feature_cost += float(self.structuralChange.cost())
        feature_cost += float(self.leftTriggers.cost())
        feature_cost += float(self.rightTriggers.cost())

        # 2. Element-based cost: count meaningful pieces once each
        def spec_elements(spec):
            # Any non-null specification (including Ø, boundaries, offsets, place)
            # counts as one element, unless it is a composite focus (e.g. metathesis).
            if spec is None:
                return 0
            if hasattr(spec, "element_count"):
                try:
                    return int(spec.element_count())
                except Exception:
                    pass
            return 1

        element_cost = 0.0
        # focus and structural change (even EmptySpecification) are elements
        element_cost += spec_elements(self.focus)
        element_cost += spec_elements(self.structuralChange)
        # guard specifications
        for s in getattr(self.leftTriggers, "specifications", []):
            if s is not None:
                element_cost += spec_elements(s)
        for s in getattr(self.rightTriggers, "specifications", []):
            if s is not None:
                element_cost += spec_elements(s)

        return feature_cost + element_cost

    def doesNothing(self):
        """Does this rule do nothing? Equivalently is it [  ] ---> [  ] /  _"""
        return (
            self.leftTriggers.doesNothing()
            and self.rightTriggers.doesNothing()
            and self.focus.doesNothing()
            and self.structuralChange.doesNothing()
        )

    def __eq__(self, o):
        return str(self) == str(o)

    def __ne__(self, o):
        return not (self == o)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.__unicode__()

    def __unicode__(self):
        if not hasattr(self, "representation") or self.representation == None:
            # check this: should I be calling Unicode recursively?
            self.representation = "{} ---> {} / {} _ {}".format(
                str(self.focus),
                str(self.structuralChange),
                str(self.leftTriggers),
                str(self.rightTriggers),
            )
        return self.representation

    def skeleton(self):
        return "{} ---> {} / {} _ {}".format(
            self.focus.skeleton(),
            self.structuralChange.skeleton(),
            self.leftTriggers.skeleton(),
            self.rightTriggers.skeleton(),
        )

    def pretty(self):
        # deletion & deGemini
        if self.isGeminiRule():
            p = str(self.rightTriggers.specifications[0]) + "ᵢ"
        else:
            p = str(self.focus)
        p += "⟶"
        # insertion and copying
        p += str(self.structuralChange)
        copyOffset = 0
        p += " / "
        p += self.leftTriggers.pretty(copyOffset)
        p += " _ "
        p += self.rightTriggers.pretty(copyOffset)
        p = p.replace("[ +vowel ]", "V")
        p = p.replace("[ -vowel ]", "C")
        p = p.replace("  ", " ")
        return p

    def calculateCopyOffset(self):
        if isinstance(self.focus, OffsetSpecification):
            return self.focus.offset
        if isinstance(self.structuralChange, OffsetSpecification):
            return self.structuralChange.offset
        return 0

    def calculateMapping(self, bank):
        # Metathesis does not define a pointwise feature mapping.
        if isinstance(self.structuralChange, MetathesisSpecification):
            return {}

        insertion = False
        deletion = isinstance(self.structuralChange, EmptySpecification)

        # construct the input/output mapping
        if isinstance(self.focus, ConstantPhoneme):
            inputs = [self.focus.p]
        elif isinstance(self.focus, FeatureMatrix):
            if not self.isGeminiRule():
                inputs = [p for p in bank.phonemes if self.focus.matches(featureMap[p])]
            else:
                inputs = [
                    p
                    for p in bank.phonemes
                    if self.rightTriggers.specifications[0].matches(featureMap[p])
                ]
        elif isinstance(self.focus, EmptySpecification):
            # insertion rule
            assert isinstance(self.structuralChange, ConstantPhoneme)
            insertion = True
            inputs = [""]
        else:
            assert False

        if deletion:
            outputs = ["" for _ in inputs]
        else:
            if not insertion:
                outputs = [
                    frozenset(self.structuralChange.apply(featureMap[i]))
                    for i in inputs
                ]
            else:
                outputs = [frozenset(featureMap[self.structuralChange.p])]
            if getVerbosity() >= 5:
                print("outputs = ", outputs)
            outputs = [bank.matrix2phoneme.get(o, None) for o in outputs]

        return {i: o for (i, o) in zip(inputs, outputs)}

    @staticmethod
    def enumeration(b, cost):
        def enumerateFocus():
            focuses = Specification.enumeration(b, cost)
            if cost > 1:
                focuses += [EmptySpecification()]
            return focuses

        def enumerateChange(focus):
            fc = focus.cost()
            isInsertion = isinstance(focus, EmptySpecification)
            if not isInsertion:
                changes = [
                    c
                    for c in Specification.enumeration(b, cost - fc)
                    if not (
                        isinstance(c, FeatureMatrix) and c.featuresAndPolarities == []
                    )
                    and not (
                        isinstance(c, ConstantPhoneme)
                        and isinstance(focus, ConstantPhoneme)
                        and c.p == focus.p
                    )
                ]
                if cost - fc > 1:
                    changes += [EmptySpecification()]
            else:
                changes = ConstantPhoneme.enumeration(b, cost - fc)
            return changes

        results = []
        for focus in enumerateFocus():
            for change in enumerateChange(focus):
                c1 = int(cost - focus.cost() - change.cost())
                for gl in Guard.enumeration("L", b, c1):
                    for gr in Guard.enumeration("R", b, int(c1 - gl.cost())):
                        if (
                            isinstance(focus, EmptySpecification)
                            and gl.doesNothing()
                            and gr.doesNothing()
                        ):
                            continue

                        results.append(Rule(focus, change, gl, gr))
        return results


EMPTYRULE = Rule(
    focus=FeatureMatrix([]),
    structuralChange=FeatureMatrix([]),
    leftTriggers=Guard("L", False, False, False, []),
    rightTriggers=Guard("R", False, False, False, []),
)
assert EMPTYRULE.doesNothing()
