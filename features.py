import unicodedata

secondStress = "secondStress"
palatal = "palatal"
palletized = "palletized"
sibilant = "sibilant"
sonorant = "sonorant"
coronal = "coronal"
retroflex = "retroflex"
creaky = "creaky"
risingTone = "risingTone"
highTone = "highTone"
lowTone = "lowTone"
middleTone = "middleTone"
longVowel = "long"
vowel = "vowel"
tense = "tense"
lax = "lax"
high = "high"
middle = "middle"
low = "low"
front = "front"
central = "central"
back = "back"
rounded = "rounded"
bilabial = "bilabial"
stop = "stop"
voice = "voice"
fricative = "fricative"
labiodental = "labiodental"
dental = "dental"
alveolar = "alveolar"
# labiovelar = "labiovelar"
velar = "velar"
nasal = "nasal"
uvular = "uvular"
glide = "glide"
liquid = "liquid"
lateral = "lateral"
trill = "trill"
flap = "flap"
affricate = "affricate"
alveopalatal = "alveopalatal"
anterior = "anterior"
aspirated = "aspirated"
unreleased = "unreleased"
laryngeal = "laryngeal"
pharyngeal = "pharyngeal"
syllableBoundary = "syllableBoundary"
wordBoundary = "wordBoundary"
continuant = "continuant"
syllabic = "syllabic"
delayedRelease = "delayedRelease"

# Not actually features...
wildFeature = "wild"
optionalFeature = "optional"

featureAbbreviation = {
    palatal: "palatal",
    palletized: "pal",
    sibilant: "sib",
    sonorant: "son",
    coronal: "cor",
    retroflex: "retro",
    creaky: "creaky",
    risingTone: "riseTn",
    highTone: "hiTn",
    lowTone: "loTn",
    middleTone: "midTn",
    longVowel: "long",
    vowel: "vowel",
    tense: "tense",
    lax: "lax",
    high: "hi",
    middle: "mid",
    low: "lo",
    front: "front",
    central: "central",
    back: "bk",
    rounded: "rnd",
    bilabial: "bilabial",
    stop: "stop",
    voice: "voice",
    fricative: "fricative",
    labiodental: "labiodental",
    dental: "dental",
    alveolar: "alveolar",
    # labiovelar:"labiovelar",
    velar: "velar",
    nasal: "nasal",
    uvular: "uvular",
    glide: "glide",
    liquid: "liq",
    lateral: "lat",
    trill: "trill",
    flap: "flap",
    affricate: "affricate",
    alveopalatal: "alveopalatal",
    anterior: "ant",
    aspirated: "asp",
    unreleased: "unreleased",
    laryngeal: "laryngeal",
    pharyngeal: "pharyngeal",
    continuant: "cont",
    syllabic: "syl",
    delayedRelease: "delRelease",
}

sophisticatedFeatureMap = {
    "*": [wildFeature],
    "?": [optionalFeature],
    # unrounded vowels
    "i": [voice, tense, high],
    "ɨ": [voice, tense, high, back],
    "ɩ": [voice, high],
    "e": [voice, tense],
    "ə": [voice, back],
    "ɛ": [voice, low],  # TODO: is this actually +low? halle seems to think so!
    "æ": [voice, low, tense],
    "a": [voice, low, tense, back],
    "ʌ": [voice, back, tense],
    # rounded vowels
    "u": [voice, tense, high, back, rounded],
    "ü": [voice, tense, high, rounded],
    "ʊ": [voice, high, back, rounded],
    "o": [voice, tense, back, rounded],
    "ö": [voice, tense, rounded],
    "ɔ": [voice, back, rounded],
    # possibly missing are umlauts
    # consonance
    "p": [
        anterior,
    ],
    "p^y": [anterior, palletized],
    "p̚": [anterior, unreleased],
    "p^h": [anterior, aspirated],
    "b": [anterior, voice],
    "b^h": [anterior, voice, aspirated],
    "f": [anterior, continuant],
    "φ": [anterior, continuant],
    "φ|": [anterior, continuant, bilabial],
    "f^y": [anterior, continuant, palletized],
    "v": [anterior, continuant, voice],
    "β": [anterior, continuant, voice],
    "β|": [anterior, continuant, voice, bilabial],
    "m": [anterior, nasal, voice, sonorant],  # continuant],
    "m̩": [anterior, nasal, voice, sonorant, syllabic],
    "m̥": [anterior, nasal, sonorant],  # ,continuant],
    "θ": [anterior, continuant, coronal],
    "d": [anterior, voice, coronal],
    # u"d̪": [voice,coronal],
    "d^z": [anterior, coronal, voice, delayedRelease],
    "d^z^h": [anterior, coronal, voice, delayedRelease, aspirated],
    "t": [anterior, coronal],
    # u"t̪": [coronal],
    "t̚": [anterior, coronal, unreleased],
    "t^s": [anterior, coronal, delayedRelease],
    "t^h": [anterior, aspirated, coronal],
    "ṭ": [anterior, retroflex, coronal],
    "ḍ": [anterior, retroflex, coronal, voice],
    "ṛ": [anterior, retroflex, coronal, voice, continuant],
    "ð": [anterior, continuant, voice, coronal],
    "z": [anterior, continuant, voice, coronal, sibilant],
    "ǰ": [voice, coronal, sibilant],  # alveopalatal,
    "ž": [continuant, voice, coronal, sibilant],  # alveopalatal,
    "ž^y": [continuant, voice, coronal, sibilant, palletized],  # alveopalatal,
    "s": [anterior, continuant, coronal, sibilant],
    "ṣ": [anterior, continuant, coronal, sibilant, retroflex],
    "n": [anterior, nasal, voice, coronal, sonorant],  # continuant],
    "n̩": [anterior, nasal, voice, coronal, syllabic, sonorant],
    "ṇ": [anterior, retroflex, nasal, voice, sonorant],  # continuant],
    "n̥": [anterior, nasal, coronal, sonorant],  # continuant],
    # conjecture: these are the same
    "ñ": [nasal, voice, coronal, sonorant],
    "n̆": [nasal, voice, coronal, sonorant],
    "ɲ": [nasal, voice, coronal, sonorant, high],
    "ɲ̩": [nasal, voice, coronal, sonorant, high, syllabic],
    "š": [continuant, coronal, sibilant],  # alveopalatal,
    "ɕ": [continuant, coronal, sibilant],  # alveolo-palatal fricative (approx.)
    "c": [palatal, coronal],  # NOT the same thing as palletized
    "ç": [continuant, palatal],
    "ɉ": [voice, palatal],
    "x̯": [palatal, coronal, continuant],
    "č": [coronal, sibilant],  # alveopalatal,
    "č^y": [coronal, sibilant, palletized],  # alveopalatal,
    "č^h": [coronal, sibilant, aspirated],  # alveopalatal,
    "k": [back, high],
    "k̚": [back, high, unreleased],
    "k^h": [back, high, aspirated],
    "k^y": [back, high, palletized],
    "x": [back, high, continuant],
    "X": [back, continuant],  # χ
    "x^y": [back, high, continuant, palletized],
    "g": [back, high, voice],
    "g^h": [back, high, voice, aspirated],
    "g^y": [back, high, voice, palletized],
    "ɣ": [back, high, continuant, voice],
    "ɣ^y": [back, high, continuant, voice, palletized],
    "ŋ": [back, high, nasal, voice, sonorant],  # continuant],
    "ŋ̩": [back, high, nasal, voice, sonorant, syllabic],
    "q": [back],
    "N": [back, nasal, voice],  # continuant],
    "G": [back, voice],
    "ʔ": [sonorant, low],  # laryngeal,
    "h": [continuant, sonorant, low],  # laryngeal,
    "ħ": [back, low, continuant, sonorant],
    # glides
    "w": [glide, voice, sonorant, continuant],
    "w̥": [glide, sonorant, continuant],
    "y": [glide, palletized, voice, sonorant, continuant],
    "j": [glide, palletized, voice, sonorant, continuant],
    # liquids
    "r": [liquid, voice, coronal, sonorant, continuant],
    "r̃": [liquid, trill, voice, coronal, sonorant, continuant],
    "r̥̃": [liquid, trill, coronal, sonorant, continuant],
    "ř": [liquid, flap, voice, coronal, sonorant, continuant],
    "l": [liquid, lateral, voice, coronal, sonorant, continuant],
    "l`": [liquid, secondStress, lateral, voice, coronal, sonorant, continuant],
    "l̥": [liquid, lateral, coronal, sonorant, continuant],
    "ʎ": [liquid, lateral, voice, palatal, sonorant, continuant],
    "ł": [liquid, lateral, voice, back, high, sonorant, continuant],
    #    u"̌l": [liquid,lateral,voice,coronal,sonorant],
    # I'm not sure what this is
    # I think it is a mistranscription, as it is in IPA but not APA
    # u"ɲ": []
    "ʕ": [back, low, voice, continuant],
    "-": [syllableBoundary],
    "##": [wordBoundary],
}

simpleFeatureMap = {
    "*": [wildFeature],
    "?": [optionalFeature],
    # unrounded vowels
    "i": [voice, tense, high, front],
    "ɨ": [voice, tense, high, back, central],
    "ɩ": [voice, high, front],
    "e": [voice, tense, middle, front],
    "ə": [voice, tense, middle, central],
    "ɛ": [voice, middle, front],
    "æ": [voice, low, front, tense],
    "a": [voice, low, central, tense],
    "ʌ": [voice, middle, central],
    # rounded vowels
    "u": [voice, tense, high, back, rounded],
    "ü": [voice, tense, high, front, rounded],
    "ʊ": [voice, high, back, rounded],
    "o": [voice, middle, tense, back, rounded],
    "ö": [voice, middle, tense, front, rounded],
    "ɔ": [voice, middle, back, rounded],
    # possibly missing are umlauts
    # consonance
    "p": [bilabial, stop],
    "p^y": [bilabial, stop, palletized],
    "p̚": [bilabial, stop, unreleased],
    "p^h": [bilabial, stop, aspirated],
    "b": [bilabial, stop, voice],
    "b^h": [bilabial, stop, voice, aspirated],
    "f": [labiodental, fricative],
    "φ": [bilabial, fricative],
    "φ|": [bilabial, fricative],
    "f^y": [labiodental, fricative, palletized],
    "v": [labiodental, fricative, voice],
    "β": [labiodental, fricative, voice],
    "β|": [bilabial, fricative, voice],
    "m": [bilabial, nasal, voice],
    "m̩": [bilabial, nasal, voice, syllabic],
    "m̥": [bilabial, nasal],  # ,continuant],
    "θ": [dental, fricative],
    "d": [alveolar, stop, voice],
    # u"d̪": [voice,coronal],
    "d^z": [alveolar, affricate, voice],
    "d^z^h": [alveolar, affricate, voice, aspirated],
    "t": [alveolar, stop],
    "t̚": [alveolar, stop, unreleased],
    "t^s": [alveolar, affricate],
    "t^h": [alveolar, stop, aspirated],
    "ṭ": [alveolar, stop, retroflex],
    "ḍ": [alveolar, stop, voice, retroflex],
    "ṛ": [alveolar, fricative, retroflex, voice],
    "ð": [dental, fricative, voice],
    "z": [fricative, voice, alveolar],
    "ǰ": [voice, alveopalatal, affricate],
    "ž": [voice, alveopalatal, fricative],
    "ž^y": [voice, alveopalatal, fricative, palletized],
    "s": [fricative, alveolar],
    "ṣ": [fricative, alveolar, retroflex],
    "n": [alveolar, nasal, voice],
    "n̩": [alveolar, nasal, voice, syllabic],
    "ṇ": [retroflex, nasal, voice],
    "n̥": [alveolar, nasal],
    "ñ": [nasal, voice, alveopalatal],
    "n̆": [nasal, voice, alveopalatal],
    "š": [fricative, alveopalatal],
    "ɕ": [fricative, alveopalatal],  # alveolo-palatal fricative
    "c": [palatal, stop],
    "ç": [fricative, palatal],
    "ɉ": [voice, palatal, stop],
    "x̯": [palatal, fricative],
    "č": [alveopalatal, affricate],
    "č^y": [alveopalatal, affricate, palletized],
    "č^h": [alveopalatal, affricate, aspirated],
    "k": [velar, stop],
    "k̚": [velar, stop, unreleased],
    "k^h": [velar, stop, aspirated],
    "k^y": [velar, stop, palletized],
    "x": [velar, fricative],
    "X": [fricative, uvular],  # χ
    "x^y": [velar, fricative, palletized],
    "g": [velar, stop, voice],
    "g^h": [velar, stop, voice, aspirated],
    "g^y": [velar, stop, voice, palletized],
    "ɣ": [velar, fricative, voice],
    "ɣ^y": [velar, fricative, voice, palletized],
    "ŋ": [velar, nasal, voice],
    "ŋ̩": [velar, nasal, voice, syllabic],
    "q": [uvular, stop],
    "N": [uvular, nasal, voice],  # continuant],
    "G": [uvular, stop, voice],
    "ʔ": [laryngeal, stop],
    "h": [laryngeal, fricative],
    "ħ": [pharyngeal, fricative],
    # glides
    "w": [glide, voice, bilabial],
    "w̥": [glide, bilabial],
    "y": [glide, voice],
    "j": [glide, palletized, voice],
    # liquids
    "r": [liquid, voice, retroflex],
    "r̃": [liquid, trill, voice, retroflex],
    "r̥̃": [liquid, trill, retroflex],
    "ř": [liquid, flap, voice, retroflex],
    "l": [liquid, lateral, voice],
    "l`": [secondStress, liquid, lateral, voice],
    "l̥": [liquid, lateral],
    "ʎ": [liquid, lateral, palatal, voice],
    "ł": [liquid, lateral, voice, velar],
    #    u"̌l": [liquid,lateral,voice,coronal,sonorant],
    # I'm not sure what this is
    # I think it is a mistranscription, as it is in IPA but not APA
    "ɲ": [nasal, voice, alveopalatal, high],
    "ɲ̩": [nasal, voice, alveopalatal, high, syllabic],
    "ʕ": [affricate, pharyngeal, voice],
    "-": [syllableBoundary],
    "##": [wordBoundary],
}

featureMap = simpleFeatureMap

_tokenizer_char_substitutions = {
    "đ": "d",
    "Đ": "d",
    "–": "-",
    "—": "-",
}


def _normalize_tokenizer_char(char):
    """Map unsupported characters to known phoneme symbols."""
    if char in _tokenizer_known_chars:
        return char
    replacement = _tokenizer_char_substitutions.get(char)
    if replacement is not None:
        return replacement
    decomposed = unicodedata.normalize("NFD", char)
    if decomposed in featureMap:
        return decomposed
    stripped = "".join(
        c for c in decomposed if unicodedata.category(c) != "Mn"
    )
    if stripped and stripped != char:
        return stripped
    if not stripped and unicodedata.category(char) == "Mn":
        return ""
    return char


def _normalize_tokenizer_input(word):
    """Normalize diacritics before tokenization."""
    if not word:
        return word
    return "".join(_normalize_tokenizer_char(char) for char in word)


def add_features_without_duplicates(feature_list, new_features):
    """
    Add features to a list without creating duplicates.
    Modifies the list in place.
    
    Args:
        feature_list: List to modify
        new_features: Features to add (can be a list, tuple, set, or single feature)
    """
    if isinstance(new_features, (list, tuple, set)):
        for f in new_features:
            if f not in feature_list:
                feature_list.append(f)
    else:
        # Single feature
        if new_features not in feature_list:
            feature_list.append(new_features)


# Automatically annotate vowels
vs = ["i", "ɨ", "ɩ", "e", "ə", "ɛ", "æ", "a", "ʌ", "u", "ü", "ʊ", "o", "ö", "ɔ"]
for fm in [simpleFeatureMap, sophisticatedFeatureMap]:
    for k in fm:
        features = fm[k]  # Use the feature map being iterated, not featureMap
        if k in vs:
            add_features_without_duplicates(features, vowel)
# feature set only apply to vowels
VOWELFEATURES = {
    rounded,
    tense,
    low,
    longVowel,
    risingTone,
    lowTone,
    highTone,
    middleTone,
}

# feature set only apply to consonants
CONSONANTFEATURES = {aspirated, palletized}

CONSONANTPLACEFEATURES = {anterior, coronal}

# features that always apply to vowels
DEFAULTVOWELFEATURES = {sonorant, continuant}

# Include vowel/consonants diacritics
vs = [k for k in featureMap if vowel in featureMap[k]]
cs = [k for k in featureMap if not (vowel in featureMap[k])]
for fm in [simpleFeatureMap, sophisticatedFeatureMap]:
    for v in vs:
        if fm == sophisticatedFeatureMap:
            add_features_without_duplicates(fm[v], DEFAULTVOWELFEATURES)
        # Create new entries without duplicates
        base_features = list(fm[v])
        add_features_without_duplicates(base_features, highTone)
        fm[v + "́"] = base_features
        base_features = list(fm[v])
        add_features_without_duplicates(base_features, lowTone)
        fm[v + "`"] = base_features
        base_features = list(fm[v])
        add_features_without_duplicates(base_features, middleTone)
        fm[v + "¯"] = base_features
        base_features = list(fm[v])
        add_features_without_duplicates(base_features, longVowel)
        fm[v + ":"] = base_features
        base_features = list(fm[v])
        add_features_without_duplicates(base_features, [longVowel, highTone])
        fm[v + "́:"] = base_features
        base_features = list(fm[v])
        add_features_without_duplicates(base_features, risingTone)
        fm[v + "̌"] = base_features
        base_features = list(fm[v])
        add_features_without_duplicates(base_features, nasal)
        fm[v + "̃"] = base_features
# Mohawk is crazy like this
v = "ʌ"
for fm in [simpleFeatureMap, sophisticatedFeatureMap]:
    base_features = list(fm[v])
    add_features_without_duplicates(base_features, [nasal, highTone])
    fm[v + "̃́"] = base_features
    base_features = list(fm[v])
    add_features_without_duplicates(base_features, [nasal, highTone, longVowel])
    fm[v + "̃́:"] = base_features


# Build tokenizer character set after all feature map augmentations
_tokenizer_known_chars = set()
for _symbol in featureMap.keys():
    _tokenizer_known_chars.update(_symbol)


# palletization
for fm in [simpleFeatureMap, sophisticatedFeatureMap]:
    for p in ["v", "b", "t", "z", "š", "l", "d", "m", "s", "t^s", "n", "r"]:
        base_features = list(fm[p])
        add_features_without_duplicates(base_features, palletized)
        fm[p + "^y"] = base_features

# Let's give sonority to the simple features also
for p, f in sophisticatedFeatureMap.items():
    if "sonorant" in f:
        add_features_without_duplicates(simpleFeatureMap[p], "sonorant")


def tokenize(word):
    originalWord = word
    word = _normalize_tokenizer_input(word)
    # FIXME: this is not part of APA, approximate with a shewha
    word = word.replace("ɜ", "ə")
    # remove all the spaces
    word = word.replace(" ", "")
    # TODO for future: support ejectives
    word = word.replace("’", "")
    # IPA > APA
    word = word.replace("ɪ", "ɩ")
    # support a-umlaut in addition to u- and o-umlauts
    word = word.replace("ä", "æ")
    # sh and zh
    word = word.replace("ʃ", "š")
    word = word.replace("ʒ", "ž")
    tokens = []
    while len(word) > 0:
        # Find the largest prefix which can be looked up in the feature dictionary
        for suffixLength in range(len(word)):
            prefixLength = len(word) - suffixLength
            prefix = word[:prefixLength]
            if prefix in featureMap:
                tokens.append(prefix)
                word = word[prefixLength:]
                break
            elif suffixLength == len(word) - 1:
                print(word)
                print(originalWord)
                raise Exception(
                    "No valid prefix: "
                    + word
                    + " when parsing "
                    + originalWord
                    + "into phonemes. Perhaps you are trying to use a phoneme that is not currently part of the system."
                )
    return tokens


class FeatureBank:
    """Builds a bank of features and sounds that are specialized to a particular data set.
    The idea is that we don't want to spend time reasoning about features/phonemes that are not attested
    """

    mutuallyExclusiveClasses = []  # ["stop","fricative","affricate"]]

    def __init__(self, words):
        self.phonemes = list(
            {
                p
                for w in words
                for p in (tokenize(w) if isinstance(w, str) else w.phonemes)
            }
        )
        self.features = list({f for p in self.phonemes for f in featureMap[p]})
        self.featureMap = {
            p: list(set(featureMap[p]) & set(self.features)) for p in self.phonemes
        }
        self.featureVectorMap = {
            p: [(f in self.featureMap[p]) for f in self.features] for p in self.phonemes
        }
        self.phoneme2index = {self.phonemes[j]: j for j in range(len(self.phonemes))}
        self.feature2index = {self.features[j]: j for j in range(len(self.features))}
        self.matrix2phoneme = {frozenset(featureMap[p]): p for p in self.phonemes}

        self.hasSyllables = syllableBoundary in self.features

    def wordToMatrix(self, w):
        return [self.featureVectorMap[p] for p in tokenize(w)]

    def __unicode__(self):
        return "FeatureBank({" + ",".join(self.phonemes) + "})"

    def __str__(self):
        return self.__unicode__()


FeatureBank.GLOBAL = FeatureBank(list(featureMap.keys()))
FeatureBank.ACTIVE = None


if __name__ == "__main__":
    vs = [k for k, v in sophisticatedFeatureMap.items() if vowel in v]

    def show(p):
        if p in ["##", "*", "-", "?"]:
            return
        print(
            p,
            "\t",
            "[ %s ]"
            % (" ".join(["+" + f for f in sorted(sophisticatedFeatureMap[p])])),
        )

    def showMany(name, ps):
        print("%s:" % name)
        for p in sorted(ps):
            show(p)
        print()

    showMany("vowels", vs)
    showMany(
        "nasals",
        [
            k
            for k, v in sophisticatedFeatureMap.items()
            if vowel not in v and nasal in v
        ],
    )
    showMany(
        "consonants",
        [
            k
            for k, v in sophisticatedFeatureMap.items()
            if vowel not in v and nasal not in v
        ],
    )
