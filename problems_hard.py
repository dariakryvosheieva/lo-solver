from solver import solve


# UKLO 2025 Foundation P1
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

# NACLO 2023 R1, problem D
permyak = [
    (
		"cerkulaɲ",
		[({"root": "house", "case": "towards"}, "N", "none")],
	),
	(
		"pɨzannezɨtlən",
		[({"root": "desk", "possessor": "your-sg", "case": "of", "number": "pl"}, "N", "none")],
	),
	(
		"ponɨt",
		[({"root": "dog", "possessor": "your-sg"}, "N", "none")],
	),
	(
		"purtnɨs",
		[({"root": "knife", "possessor": "their"}, "N", "none")],
	),
	(
		"kəinnezɨs",
		[({"root": "wolf", "possessor": "his", "number": "pl"}, "N", "none")],
	),
	(
		"vərələn",
		[({"root": "forest", "case": "of", "possessor": "my"}, "N", "none")],
	),
	(
		"purtəla",
		[({"root": "knife", "case": "for the sake of", "possessor": "my"}, "N", "none")],
	),
	(
		"tɨezɨtkət",
		[({"root": "lake", "case": "with", "possessor": "your-sg", "number": "pl"}, "N", "none")],
	),
	(
		"cerkuezliɕ",
		[({"root": "house", "case": "from", "number": "pl"}, "N", "none")],
	),
	(
		"juɕɕezə",
		[({"root": "swan", "possessor": "my", "number": "pl"}, "N", "none")],
	),
	(
		"kokɨskət",
		[({"root": "foot", "case": "with", "possessor": "his"}, "N", "none")],
	),
	(
		"ciɨtlaɲ",
		[({"root": "hand", "case": "towards", "possessor": "your-sg"}, "N", "none")],
	),
	(
		"pɨzanɨsliɕ",
		[({"root": "desk", "case": "from", "possessor": "his"}, "N", "none")],
	),
	(
		"vərrezlən",
		[({"root": "forest", "case": "of", "number": "pl"}, "N", "none")],
	),
	(
		"ponnɨt",
		[({"root": "dog", "possessor": "your-pl"}, "N", "none")],
	),
	(
		"juɕla",
		[({"root": "swan", "case": "for the sake of"}, "N", "none")],
    ),
]

# IOL 2008 P4
zoque = [
	(
		"mis nakpatpit",
		[({"root": "you"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "cactus", "case": "with"}, "N", "none")],
	),
	(
		"nakpat",
		[({"root": "cactus"}, "N", "none")],
	),
	(
		"mokpittih",
		[({"root": "corn", "case": "with", "only": "only"}, "N", "none")],
	),
	(
		"pokskukyʌsmʌtaʔm",
		[({"root": "chair", "case": "above", "number": "pl"}, "N", "none")],
	),
	(
		"pokskuy",
		[({"root": "chair"}, "N", "none")],
	),
	(
		"peroltih",
		[({"root": "kettle", "only": "only"}, "N", "none")],
	),
	(
		"kot^sʌktaʔm",
		[({"root": "mountain", "number": "pl"}, "N", "none")],
	),
	(
		"komgʌsmʌtih",
		[({"root": "post", "case": "above", "only": "only"}, "N", "none")],
	),
	(
		"ʔʌs ŋgom",
		[({"root": "I"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "post"}, "N", "none")],
	),
	(
		"kʌmʌŋbitšeh",
		[({"root": "shadow", "case": "with", "like": "like"}, "N", "none")],
	),
	(
		"kʌmʌŋdaʔm",
		[({"root": "shadow", "number": "pl"}, "N", "none")],
	),
	(
		"ʔʌs nd^zapkʌsmʌšeh",
		[({"root": "I"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "sky", "case": "above", "like": "like"}, "N", "none")],
	),
	(
		"t^sapšeh",
		[({"root": "sky", "like": "like"}, "N", "none")],
	),
	(
		"pahsungotoya",
		[({"root": "squash", "case": "for"}, "N", "none")],
	),
	(
		"pahsunšehtaʔmdih",
		[({"root": "squash", "like": "like", "number": "pl", "only": "only"}, "N", "none")],
	),
	(
		"tʌt^skotoyatih",
		[({"root": "tooth", "case": "for", "only": "only"}, "N", "none")],
	),
	(
		"kumgukyʌsmʌ",
		[({"root": "town", "case": "above"}, "N", "none")],
	),
	(
		"kumgukyotoyataʔm",
		[({"root": "town", "case": "for", "number": "pl"}, "N", "none")],
	),
	(
		"t^sakyotoya",
		[({"root": "vine", "case": "for"}, "N", "none")],
	),
	(
		"mis nd^zay",
		[({"root": "you"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "vine"}, "N", "none")],
	),
	(
		"t^sakyʌsmʌtih",
		[({"root": "vine", "case": "above", "only": "only"}, "N", "none")],
	),
	(
		"kʌmʌŋšeh",
		[({"root": "shadow", "like": "like"}, "N", "none")],
	),
	(
		"ʔʌs mok",
		[({"root": "I"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "corn"}, "N", "none")],
	),
	(
		"mis ndʌt^staʔm",
		[({"root": "you"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "tooth", "number": "pl"}, "N", "none")],
	),
	(
		"pahsunbit",
		[({"root": "squash", "case": "with"}, "N", "none")],
	),
	(
		"perolkotoyašehtaʔm",
		[({"root": "kettle", "case": "for", "like": "like", "number": "pl"}, "N", "none")],
	),
]


if __name__ == "__main__":
	solve(fur, type="rosetta")
	solve(permyak, type="rosetta")
	solve(zoque, type="rosetta")