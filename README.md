# LO Solver v0
This is the beta-release of the **LO Solver**, a symbolic solver for linguistics olympiad problems.

Building upon my final project for Prof. Solar-Lezama's [program synthesis](https://people.csail.mit.edu/asolar/SynthesisCourse/index.htm) class at MIT, the solver uses an approach inspired by symbolic program synthesis to discover solutions to LO problems. The solving process is modeled as a heuristic search over all candidate solutions.

This work has been influenced by the paper *[Synthesizing theories of human language with Bayesian program induction](https://www.nature.com/articles/s41467-022-32012-w)* (*Nature*, 2022); many features are borrowed directly from the associated [codebase](https://github.com/ellisk42/bpl_phonology).

## Setup

The solver does not use any third-party Python libraries.

1. Clone the repo:
```bash
git clone https://github.com/dariakryvosheieva/lo-solver.git
```
2. Try it:
```bash
python problems_easy.py
```

## Problem Input Format

The input is a list of `(sentence, translation)` tuples where:
* `sentence`: raw string in Problemese (= target language)
* `translation`: list of `(features, pos, role)` tuples representing words
    * `features`: dict of morphological features (e.g., `{'root': 'problem', 'number': 'pl'}`)
	* `pos`: part of speech (e.g., 'N' for noun)
	* `role`: syntactic role (e.g., 'S' for subject)

<details>
<summary>Example (click to expand)</summary>

```python
from solver import solve

fur = [
	(
		"Kòrò jabèl.",
		[
			({"subject": "you", "root": "drink"}, "V", "V"),
			({"root": "water"}, "N", "O"),
		],
	),
	(
		"Yáà tòn tuumèl.",
		[
			({"root": "woman"}, "N", "S"),
			({"subject": "he/she/it", "root": "build"}, "V", "V"),
			({"root": "house"}, "N", "O"),
		],
	),
	(
		"Bàin kòrò lemèl.",
		[
			({"root": "elder"}, "N", "S"),
			({"subject": "he/she/it", "root": "lick"}, "V", "V"),
			({"root": "water"}, "N", "O"),
		],
	),
	(
		"Tònà jutumèl.",
		[
			({"subject": "you", "root": "build"}, "V", "V"),
			({"root": "house", "number": "pl"}, "N", "O"),
		],
	),
	(
		"Kàrabà kòrò kelmèlà.",
		[
			({"root": "animal", "number": "pl"}, "N", "S"),
			({"subject": "they", "root": "lick", "number": "pl"}, "V", "V"),
			({"root": "water"}, "N", "O"),
		],
	),
	(
		"Tòn yáà bauèl.",
		[
			({"root": "house"}, "N", "S"),
			({"subject": "he/she/it", "root": "hold"}, "V", "V"),
			({"root": "woman"}, "N", "O"),
		],
	),
	(
		"Bàinà tòn kutumèlà.",
		[
			({"root": "elder", "number": "pl"}, "N", "S"),
			({"subject": "they", "root": "build", "number": "pl"}, "V", "V"),
			({"root": "house"}, "N", "O"),
		],
	),
	(
		"Yáà dèl baèl.",
		[
			({"root": "woman"}, "N", "S"),
			({"subject": "he/she/it", "root": "drink"}, "V", "V"),
			({"root": "oil"}, "N", "O"),
		],
	),
	(
		"Jirgèl.",
		[
			({"subject": "you", "root": "tie"}, "V", "V"),
		],
	),
	(
		"Yáà bàin ruñèl.",
		[
			({"root": "woman"}, "N", "S"),
			({"subject": "he/she/it", "root": "lift"}, "V", "V"),
			({"root": "elder"}, "N", "O"),
		],
	),
	(
		"Kòrò jelmèl.",
		[
			({"subject": "you", "root": "lick"}, "V", "V"),
			({"root": "water"}, "N", "O"),
		],
	),
	(
		"Yáà rigèl.",
		[
			({"root": "woman"}, "N", "S"),
			({"subject": "he/she/it", "root": "tie"}, "V", "V"),
		],
	),
]

solve(fur, type="rosetta")

# Output:
# Word order: S - O - V
# Morpheme order (N): root - number
# Morpheme order (V): subject - root - number
# Vocabulary:
#   (a, pl)
#   (bael, drink)
#   (bain, elder)
#   (bauel, hold)
#   (del, oil)
#   (j, you)
#   (k, they)
#   (karab, animal)
#   (koro, water)
#   (lemel, lick)
#   (rigel, tie)
#   (ruñel, lift)
#   (ton, house)
#   (tuumel, build)
#   (yáa, woman)
# Rules:
# 1 2⟶MET / [ +glide ] _ 
# 1 2⟶MET / k _ 
# Time taken: 893.75 s
```

</details>

<br>

See `problems_easy.py` and `problems_hard.py` for more examples.

## Technical Details

### Runtime

Most problems run faster than the time it would take a human expert to solve them (from fractions of a second to ~25 minutes for IOL-level problems).

### Solving Strategy

The following algorithm is applied to each problem:

1. Determine the word order and the English-to-Problemese word mappings.
2. For each part of speech:

    1. Sort candidate morpheme orders by how likely they are to be correct.

    2. For each candidate morpheme order:

        1. Find all plausible segmentations of Problemese words into morphemes.

        2. For each plausible segmentation, find a minimal set of phonological rules that explains all differences in morpheme variants. As soon as such a set is found, go to step 3.
3. Aggregate phonological rules across parts of speech and return the final output.

## Limitations

### Fundamental

* At present, the solver more closely resembles an **aligner between Problemese sentences and pre-specified glosses** than a fully end-to-end linguistic data analysis system. That is, it operates on raw Problemese sentences and *annotated* English translations, rather than both in raw form. This is less of an issue if the Problemese sentences use the same grammatical categories as the English translations, as it suffices to analyze the latter. However, if Problemese uses categories absent from English, such as cases, noun classes etc., the user should discover this and provide appropriate annotations.
* LO problems usually involve **additional tasks beyond the analysis of the data provided in the main statement**, such as English-to-Problemese and Problemese-to-English translation tasks. The solver cannot handle those, but hopefully the user will be able to handle them more easily after obtaining the lexicon and rules from the solver.

### Future Work

* Currently, only Rosetta Stone problems are supported. In the near future, I plan to add support for counting/numeral systems and semantic matching (insofar as an algorithm can capture the complexity of semantic compounding).
* Other features I still need to implement: syntactic constituents, vowel harmony, ejectives, word boundary guards.

## Intended Use

The solver is intended to help students prepare for LOs, and as a proof of concept that LO problems can be solved automatically. Needless to say, it is *not* intended for cheating in online contests.