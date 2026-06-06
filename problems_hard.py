from solver import solve


### ROSETTA ###

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

# NACLO 2018 R2, problem O
tamil = [
	(
		"paṭittēn",
		[
			({"subject": "I", "root": "learn", "tense": "PST"}, "V", "V"),
		],
	),
	(
		"paṭippīr",
		[
			({"subject": "you-pl", "root": "learn", "tense": "FUT"}, "V", "V"),
		],
	),
	(
		"ceyyān",
		[
			({"subject": "he", "root": "do", "tense": "NEG"}, "V", "V"),
		],
	),
	(
		"paṭiyēn",
		[
			({"subject": "I", "root": "learn", "tense": "NEG"}, "V", "V"),
		],
	),
	(
		"arippāy",
		[
			({"subject": "you-sg", "root": "know", "tense": "FUT"}, "V", "V"),
		],
	),
	(
		"paṭittāy",
		[
			({"subject": "you-sg", "root": "learn", "tense": "PST"}, "V", "V"),
		],
	),
	(
		"arittān",
		[
			({"subject": "he", "root": "know", "tense": "PST"}, "V", "V"),
		],
	),
	(
		"aarambippāy",
		[
			({"subject": "you-sg", "root": "begin", "tense": "FUT"}, "V", "V"),
		],
	),
	(
		"aarambippōm",
		[
			({"subject": "we", "root": "begin", "tense": "FUT"}, "V", "V"),
		],
	),
	(
		"aarambiyēn",
		[
			({"subject": "I", "root": "begin", "tense": "NEG"}, "V", "V"),
		],
	),
	(
		"paṭittāl",
		[
			({"subject": "she", "root": "learn", "tense": "PST"}, "V", "V"),
		],
	),
	(
		"ceyppēn",
		[
			({"subject": "I", "root": "do", "tense": "FUT"}, "V", "V"),
		],
	),
	(
		"ceyyēn",
		[
			({"subject": "I", "root": "do", "tense": "NEG"}, "V", "V"),
		],	
	),
	(
		"ceyppār",
		[
			({"subject": "they", "root": "do", "tense": "FUT"}, "V", "V"),
		],
	),
]

# IOL 2008 P4
zoque = [
	(
		"mis nakpatpit",
		[({"root": "you"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "cactus", "case": "with"}, "N", "N")],
	),
	(
		"nakpat",
		[({"root": "cactus"}, "N", "N")],
	),
	(
		"mokpittih",
		[({"root": "corn", "case": "with", "only": "only"}, "N", "N")],
	),
	(
		"pokskukyʌsmʌtaʔm",
		[({"root": "chair", "case": "above", "number": "pl"}, "N", "N")],
	),
	(
		"pokskuy",
		[({"root": "chair"}, "N", "N")],
	),
	(
		"peroltih",
		[({"root": "kettle", "only": "only"}, "N", "N")],
	),
	(
		"kot^sʌktaʔm",
		[({"root": "mountain", "number": "pl"}, "N", "N")],
	),
	(
		"komgʌsmʌtih",
		[({"root": "post", "case": "above", "only": "only"}, "N", "N")],
	),
	(
		"ʔʌs ŋgom",
		[({"root": "I"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "post"}, "N", "N")],
	),
	(
		"kʌmʌŋbitšeh",
		[({"root": "shadow", "case": "with", "like": "like"}, "N", "N")],
	),
	(
		"kʌmʌŋdaʔm",
		[({"root": "shadow", "number": "pl"}, "N", "N")],
	),
	(
		"ʔʌs nd^zapkʌsmʌšeh",
		[({"root": "I"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "sky", "case": "above", "like": "like"}, "N", "N")],
	),
	(
		"t^sapšeh",
		[({"root": "sky", "like": "like"}, "N", "N")],
	),
	(
		"pahsungotoya",
		[({"root": "squash", "case": "for"}, "N", "N")],
	),
	(
		"pahsunšehtaʔmdih",
		[({"root": "squash", "like": "like", "number": "pl", "only": "only"}, "N", "N")],
	),
	(
		"tʌt^skotoyatih",
		[({"root": "tooth", "case": "for", "only": "only"}, "N", "N")],
	),
	(
		"kumgukyʌsmʌ",
		[({"root": "town", "case": "above"}, "N", "N")],
	),
	(
		"kumgukyotoyataʔm",
		[({"root": "town", "case": "for", "number": "pl"}, "N", "N")],
	),
	(
		"t^sakyotoya",
		[({"root": "vine", "case": "for"}, "N", "N")],
	),
	(
		"mis nd^zay",
		[({"root": "you"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "vine"}, "N", "N")],
	),
	(
		"t^sakyʌsmʌtih",
		[({"root": "vine", "case": "above", "only": "only"}, "N", "N")],
	),
	(
		"kʌmʌŋšeh",
		[({"root": "shadow", "like": "like"}, "N", "N")],
	),
	(
		"ʔʌs mok",
		[({"root": "I"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "corn"}, "N", "N")],
	),
	(
		"mis ndʌt^staʔm",
		[({"root": "you"}, "PRN", "Poss"), ({"possessive": "possessive", "root": "tooth", "number": "pl"}, "N", "N")],
	),
	(
		"pahsunbit",
		[({"root": "squash", "case": "with"}, "N", "N")],
	),
	(
		"perolkotoyašehtaʔm",
		[({"root": "kettle", "case": "for", "like": "like", "number": "pl"}, "N", "N")],
	),
]


### SCRAMBLED ROSETTA ###

# UKLO 2014 R2 P1
swahili = [
	[
		"Alikula",
		"Atacheza",
		"Mlifahamu",
		"Mnapika",
		"Nilicheza",
		"Ninakula",
		"Ninapika",
		"Nitapika",
		"Tulifahamu",
		"Unacheza",
		"Utapika",
		"Wanafahamu",
		"Watapika",
		"Walicheza",
	],
	[
		[({"root": "eat", "subject": "he/she", "tense": "PST"}, "V", "V")],
		[({"root": "play", "subject": "he/she", "tense": "FUT"}, "V", "V")],
		[({"root": "eat", "subject": "I", "tense": "PRS"}, "V", "V")],
		[({"root": "play", "subject": "I", "tense": "PST"}, "V", "V")],
		[({"root": "cook", "subject": "I", "tense": "PRS"}, "V", "V")],
		[({"root": "cook", "subject": "I", "tense": "FUT"}, "V", "V")],
		[({"root": "understand", "subject": "they", "tense": "PRS"}, "V", "V")],
		[({"root": "cook", "subject": "they", "tense": "FUT"}, "V", "V")],
		[({"root": "play", "subject": "they", "tense": "PST"}, "V", "V")],
		[({"root": "understand", "subject": "we", "tense": "PST"}, "V", "V")],
		[({"root": "understand", "subject": "you-pl", "tense": "PST"}, "V", "V")],
		[({"root": "cook", "subject": "you-pl", "tense": "PRS"}, "V", "V")],
		[({"root": "play", "subject": "you-sg", "tense": "PRS"}, "V", "V")],
		[({"root": "cook", "subject": "you-sg", "tense": "FUT"}, "V", "V")],
	],
]

# NACLO 2010 R1, problem G
tangkhul = [
	[
		"a masikserra",
		"āni masikngarokei",
		"āthum masikngarokngāilā",
		"ini thāingarokei",
		"na thāilā",
		"ithum thāingāihāirara",
		"rāserhāira",
		"āni rāra",
		"nathum rāserhāiralā",
	],
	[
		[
			({"root": "3", "number": "pl"}, "PRN", "S"),
			({"root": "pinch", "tense": "Q (PRS)", "want": "want", "RECP": "RECP"}, "V", "V"),
		],
		[
			({"root": "2", "number": "sg"}, "PRN", "S"),
			({"root": "see", "tense": "Q (PRS)"}, "V", "V"),
		],
		[
			({"root": "2", "number": "pl"}, "PRN", "S"),
			({"root": "come", "tense": "Q (PRS)", "aspect": "PFV", "all": "all"}, "V", "V"),
		],
		[
			({"root": "3", "number": "sg"}, "PRN", "S"),
			({"root": "pinch", "tense": "FUT", "all": "all"}, "V", "V"),
		],
		[
			({"root": "come", "aspect": "PFV", "all": "all"}, "V", "V"),
		],
		[
			({"root": "3", "number": "du"}, "PRN", "S"),
			({"root": "pinch", "tense": "PST", "RECP": "RECP"}, "V", "V"),
		],
		[
			({"root": "3", "number": "du"}, "PRN", "S"),
			({"root": "come", "tense": "FUT"}, "V", "V"),
		],
		[
			({"root": "1", "number": "pl"}, "PRN", "S"),
			({"root": "see", "tense": "FUT", "aspect": "PFV", "want": "want"}, "V", "V"),
		],
		[
			({"root": "1", "number": "du"}, "PRN", "S"),
			({"root": "see", "tense": "PST", "RECP": "RECP"}, "V", "V"),
		],
	],
]


### SEMANTIC MATCHING ###

# IOL 2017 P2
abui = [
	[
		"abang",
		"atáng heya",
		"bataa hawata",
		"dekafi",
		"ebataa hatáng",
		"ekuda hawata",
		"falepak hawei",
		"hatáng hamin",
		"helui",
		"maama hefalepak",
		"napong",
		"rièng",
		"ritama",
		"riya hatáng",
		"tama habang",
		"tamin",
		"tefe hawei",
	],
	[
		[
			({"possessor": "3sg", "root": "hand"}, "N", "possessor"),
			({"possessor": "3sg", "root": "nose"}, "N", "possessed"),
		],
		[
			({"possession-type": "alienable", "root": "tree"}, "N", "possessor"),
			({"possessor": "3sg", "root": "hand"}, "N", "possessed"),
		],
		[
			({"possessor": "1sg", "root": "face"}, "N", "possessed"),
		],
		[
			({"possessor": "one's own", "possession-type": "alienable", "root": "rope"}, "N", "possessed"),
		],
		[
			({"root": "shoulder"}, "N", "possessed"),
		],
		[
			({"possessor": "2pl", "root": "mother"}, "N", "possessor"),
			({"possessor": "3sg", "root": "hand"}, "N", "possessed"),
		],
		[
			({"possessor": "1pl", "possession-type": "alienable", "root": "pig"}, "N", "possessor"),
			({"possessor": "3sg", "root": "ear"}, "N", "possessed"),
		],
		[
			({"root": "father"}, "N", "possessor"),
			({"possessor": "3sg", "possession-type": "alienable", "root": "pistol"}, "N", "possessed"),
		],
		[
			({"possession-type": "alienable", "root": "horse"}, "N", "possessor"),
			({"possessor": "3sg", "root": "neck"}, "N", "possessed"),
		],
		[
			({"root": "pistol"}, "N", "possessor"),
			({"possessor": "3sg", "root": "ear"}, "N", "possessed"),
		],
		[
			({"possessor": "2pl", "root": "eyes"}, "N", "possessed"),
		],
		[
			({"possessor": "1pl", "root": "nose"}, "N", "possessed"),
		],
		[
			({"possessor": "3sg", "possession-type": "alienable", "root": "knife"}, "N", "possessed"),
		],
		[
			({"root": "sea"}, "N", "possessor"),
			({"possessor": "3sg", "root": "shoulder"}, "N", "possessed"),
		],
		[
			({"root": "tree"}, "N", "possessor"),
			({"possessor": "3sg", "root": "neck"}, "N", "possessed"),
		],
		[
			({"root": "hand"}, "N", "possessor"),
			({"possessor": "3sg", "possession-type": "alienable", "root": "mother"}, "N", "possessed"),
		],
		[
			({"possessor": "2pl", "root": "sea"}, "N", "possessed"),
		],
	],
]


if __name__ == "__main__":
	solve(fur, type="rosetta")
	solve(permyak, type="rosetta")
	solve(tamil, type="rosetta")
	solve(zoque, type="rosetta")
	solve(swahili, type="scrambled_rosetta")
	solve(tangkhul, type="scrambled_rosetta")
	solve(abui, type="scrambled_rosetta")