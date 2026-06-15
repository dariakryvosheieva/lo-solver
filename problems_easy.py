from solver import solve


### ROSETTA ###

# Ukraine LO 2023/24, qualification round, middle school division, P1
mizo = [
	(
		"in ip kan phur",
		[
			({"root": "we"}, "PRN", "S"),
			({"root": "carry"}, "V", "V"),
			({"root": "you-pl"}, "PRN", "Poss"),
			({"root": "bag"}, "N", "O")
		],
	),
	(
		"i ui a suu",
		[
			({"root": "he"}, "PRN", "S"),
			({"root": "wash"}, "V", "V"),
			({"root": "you-sg"}, "PRN", "Poss"),
			({"root": "dog"}, "N", "O")
		],
	),
	(
		"in aar i uum",
		[
			({"root": "you-sg"}, "PRN", "S"),
			({"root": "chase"}, "V", "V"),
			({"root": "you-pl"}, "PRN", "Poss"),
			({"root": "chicken"}, "N", "O")
		],
	),
	(
		"a vok in vuaa",
		[
			({"root": "you-pl"}, "PRN", "S"),
			({"root": "beat"}, "V", "V"),
			({"root": "he"}, "PRN", "Poss"),
			({"root": "pig"}, "N", "O")
		],
	),
	(
		"kan ui a uum",
		[
			({"root": "he"}, "PRN", "S"),
			({"root": "chase"}, "V", "V"),
			({"root": "we"}, "PRN", "Poss"),
			({"root": "dog"}, "N", "O")
		],
	),
    (
        "a in kan suu",
        [
			({"root": "we"}, "PRN", "S"),
			({"root": "wash"}, "V", "V"),
			({"root": "he"}, "PRN", "Poss"),
			({"root": "house"}, "N", "O")
		],
    ),
]

# NACLO 2008 R1, problem A
apinaye = [
	(
		"Kukrɛ̃ kokoi.",
		[
			({"root": "monkey"}, "N", "S"),
			({"root": "eat"}, "V", "V"),
		],
	),
	(
		"Ape kra.",
		[
			({"root": "child"}, "N", "S"),
			({"root": "work"}, "V", "V"),
		],
	),
	(
		"Ape kokoi ratš.",
		[
			({"root": "big"}, "Adj", "Adj"),
			({"root": "monkey"}, "N", "S"),
			({"root": "work"}, "V", "V"),
		],
	),
	(
		"Ape mï mɛtš.",
		[
			({"root": "good"}, "Adj", "Adj"),
			({"root": "man"}, "N", "S"),
			({"root": "work"}, "V", "V"),
		],
	),
	(
		"Ape mɛtš kra.",
		[
			({"root": "child"}, "N", "S"),
			({"root": "work"}, "V", "V"),
			({"root": "good"}, "Adv", "Adv"),
		],
	),
	(
		"Ape punui mï piŋɛtš.",
		[
			({"root": "old"}, "Adj", "Adj"),
			({"root": "man"}, "N", "S"),
			({"root": "work"}, "V", "V"),
			({"root": "bad"}, "Adv", "Adv"),
		],
	),
]

# UKLO 2016 Advanced P5
nung = [
	(
		"Cáu ca vửhn-nhahng kíhn.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "be about to"}, "be about to", "be about to"),
			({"root": "continue"}, "continue", "continue"),
			({"root": "eat"}, "V", "V"),
		],
	),
	(
		"Cáu cháhn slờng páy mi?",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "truly"}, "truly", "truly"),
			({"root": "want"}, "modal", "modal"),
			({"root": "go"}, "V", "V"),
			({"root": "question particle"}, "Q", "Q"),
		],
	),
	(
		"Cáu mi slày kíhn.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "not"}, "not", "not"),
			({"root": "not have to"}, "modal", "modal"),
			({"root": "eat"}, "V", "V"),
		],
	),
	(
		"Cáu ngám hẻht pehn-tế.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "just now"}, "T", "T"),
			({"root": "do"}, "V", "V"),
			({"root": "like that"}, "like that", "like that"),
		],
	),
	(
		"Cáu tan-đohc hảhn mưhng.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "only"}, "only", "only"),
			({"root": "see"}, "V", "V"),
			({"root": "you"}, "PRN", "O"),
		],
	),
	(
		"Cáu vửhn-nhahng bô-sạhm tảhng hẻht hơn.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "continue"}, "continue", "continue"),
			({"root": "also"}, "also", "also"),
			({"root": "do"}, "V", "V"),
			({"root": "house"}, "N", "O"),
			({"root": "alone"}, "alone", "alone"),
		],
	),
	(
		"Da kíhn!",
		[
			({"root": "don't"}, "don't", "don't"),
			({"root": "eat"}, "V", "V"),
		],
	),
	(
		"Da khải hơn!",
		[
			({"root": "don't"}, "don't", "don't"),
			({"root": "sell"}, "V", "V"),
			({"root": "house"}, "N", "O"),
		],
	),
	(
		"Mưhn chớng ca cháhn fải khải.",
		[
			({"root": "she"}, "PRN", "S"),
			({"root": "then"}, "then", "then"),
			({"root": "be about to"}, "be about to", "be about to"),
			({"root": "truly"}, "truly", "truly"),
			({"root": "have to"}, "modal", "modal"),
			({"root": "sell"}, "V", "V"),
		],
	),
	(
		"Mưhn mi cháhn đày non.",
		[
			({"root": "she"}, "PRN", "S"),
			({"root": "not"}, "not", "not"),
			({"root": "truly"}, "truly", "truly"),
			({"root": "can"}, "modal", "modal"),
			({"root": "sleep"}, "V", "V"),
		],
	),
	(
		"Mưhn náhc-thày chớng bô-sạhm kíhn.",
		[
			({"root": "she"}, "PRN", "S"),
			({"root": "just previously"}, "T", "T"),
			({"root": "then"}, "then", "then"),
			({"root": "also"}, "also", "also"),
			({"root": "eat"}, "V", "V"),
		],
	),
	(
		"Mưhng náhc-thày slờng tảhng páy.",
		[
			({"root": "you"}, "PRN", "S"),
			({"root": "just previously"}, "T", "T"),
			({"root": "want"}, "modal", "modal"),
			({"root": "go"}, "V", "V"),
			({"root": "alone"}, "alone", "alone"),
		],
	),
	(
		"Cáu cháhn đày non.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "truly"}, "truly", "truly"),
			({"root": "can"}, "modal", "modal"),
			({"root": "sleep"}, "V", "V"),
		],
	),
	(
		"Mưhn bô-sạhm mi slờng hẻht hơn mi?",
		[
			({"root": "she"}, "PRN", "S"),
			({"root": "also"}, "also", "also"),
			({"root": "not"}, "not", "not"),
			({"root": "want"}, "modal", "modal"),
			({"root": "do"}, "V", "V"),
			({"root": "house"}, "N", "O"),
			({"root": "question particle"}, "Q", "Q"),
		],
	),
	(
		"Mưhn ngám bô-sạhm páy hơn.",
		[
			({"root": "she"}, "PRN", "S"),
			({"root": "also"}, "also", "also"),
			({"root": "go"}, "V", "V"),
			({"root": "house"}, "N", "O"),
			({"root": "just now"}, "T", "T"),
		],
	),
]

# UKLO 2016 Intermediate P2
amele = [
	(
		"Naus ho uten.",
		[
			({"root": "Naus"}, "N", "S"),
			({"root": "give", "subject": "he", "indirect object": "him"}, "V", "V"),
			({"root": "pig"}, "N", "DO"),
		]
	),
	(
		"Ija dana-leis jo ihacaliga.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "show", "subject": "I", "indirect object": "them-du"}, "V", "V"),
			({"root": "two men"}, "N", "IO"),
			({"root": "house"}, "N", "DO"),
		]
	),
	(
		"Uqa sab jen.",
		[
			({"root": "he"}, "PRN", "S"),
			({"root": "eat", "subject": "he"}, "V", "V"),
			({"root": "food"}, "N", "DO"),
		]
	),
	(
		"Ele sab jowa.",
		[
			({"root": "we-du"}, "PRN", "S"),
			({"root": "eat", "subject": "we-du"}, "V", "V"),
			({"root": "food"}, "N", "DO"),
		]
	),
	(
		"Ija sab qetaliga.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "cut", "subject": "I", "indirect object": "them-du"}, "V", "V"),
			({"root": "food"}, "N", "DO"),
		]
	),
	(
		"Uqa bagol iten.",
		[
			({"root": "he"}, "PRN", "S"),
			({"root": "give", "subject": "he", "indirect object": "me"}, "V", "V"),
			({"root": "present"}, "N", "DO"),
		]
	),
	(
		"Ija sab utiga.",
		[
			({"root": "I"}, "PRN", "S"),
			({"root": "give", "subject": "I", "indirect object": "him"}, "V", "V"),
			({"root": "food"}, "N", "DO"),
		]
	),
	(
		"Uqa jo ihacuten.",
		[
			({"root": "he"}, "PRN", "S"),
			({"root": "show", "subject": "he", "indirect object": "him"}, "V", "V"),
			({"root": "house"}, "N", "DO"),
		]
	),
	(
		"Ele ho adowa.",
		[
			({"root": "we-du"}, "PRN", "S"),
			({"root": "give", "subject": "we-du", "indirect object": "you-du"}, "V", "V"),
			({"root": "pig"}, "N", "DO"),
		]
	),
	(
		"Jo ihacitaga.",
		[
			({"root": "show", "subject": "IMP", "indirect object": "me"}, "V", "V"),
			({"root": "house"}, "N", "DO"),
		]
	),
	(
		"Sab qetalaga.",
		[
			({"root": "cut", "subject": "IMP", "indirect object": "them-du"}, "V", "V"),
			({"root": "food"}, "N", "DO"),
		]
	),
]

# TOL 2026 R1 P1
arrernte = [
	(
		"aherre apetyeme",
		[
			({"root": "kangaroo"}, "N", "S"),
			({"root": "come", "tense": "PRS"}, "V", "V"),
		],
	),
	(
		"aherrele arlewatyerre atweke",
		[
			({"root": "kangaroo", "case": "ERG"}, "N", "S"),
			({"root": "hit", "tense": "PST"}, "V", "V"),
			({"root": "goanna"}, "N", "O"),
		],
	),
	(
		"akngwelyele arleye aretyenhe",
		[
			({"root": "dog", "case": "ERG"}, "N", "S"),
			({"root": "look", "tense": "FUT"}, "V", "V"),
			({"root": "emu"}, "N", "O"),
		],
	),
	(
		"ampele intelyapelyape areke",
		[
			({"root": "child", "case": "ERG"}, "N", "S"),
			({"root": "look", "tense": "PST"}, "V", "V"),
			({"root": "butterfly"}, "N", "O"),
		],
	),
	(
		"arleye alkereke-ireme",
		[
			({"root": "emu"}, "N", "S"),
			({"root": "fly", "tense": "PRS"}, "V", "V"),
		],
	),
	(
		"artwele ampe akaltyele-anteme",
		[
			({"root": "man", "case": "ERG"}, "N", "S"),
			({"root": "teach", "tense": "PRS"}, "V", "V"),
			({"root": "child"}, "N", "O"),
		],
	),
	(
		"marle apetyetyenhe",
		[
			({"root": "girl"}, "N", "S"),
			({"root": "come", "tense": "FUT"}, "V", "V"),
		],
	),
	(
		"irretye alkereke-ireke",
		[
			({"root": "wedge-tailed eagle"}, "N", "S"),
			({"root": "fly", "tense": "PST"}, "V", "V"),
		],
	),
]

# UKLO 2013 Foundation P2
zapotec = [
	(
		"nee",
		[({"root": "foot"}, "N", "none")],
	),
	(
		"kaʒikebe",
		[({"root": "shoulder", "possessor": "his", "number": "pl"}, "N", "none")],
	),
	(
		"neeluʔ",
		[({"root": "foot", "possessor": "your-sg"}, "N", "none")],
	),
	(
		"kaʒigitu",
		[({"root": "chin", "possessor": "your-pl", "number": "pl"}, "N", "none")],
	),
	(
		"ʒike",
		[({"root": "shoulder"}, "N", "none")],
	),
	(
		"biʃoʒedu",
		[({"root": "father", "possessor": "our"}, "N", "none")],
	),
	(
		"kaneebe",
		[({"root": "foot", "possessor": "his", "number": "pl"}, "N", "none")],
	),
	(
		"kabiʃoʒedu",
		[({"root": "father", "possessor": "our", "number": "pl"}, "N", "none")],
	),
	(
		"kaneetu",
		[({"root": "foot", "possessor": "your-pl", "number": "pl"}, "N", "none")],
	),
	(
		"biʃoʒeluʔ",
		[({"root": "father", "possessor": "your-sg"}, "N", "none")],
	),
	(
		"kaʒiketu",
		[({"root": "shoulder", "possessor": "your-pl", "number": "pl"}, "N", "none")],
	),
]

# PLO 2019 R1 P7
kove = [
	(
		"yau ngaoli vula pa tuanga",
		[
			({"root": "I (S)"}, "PRN", "S"),
			({"root": "buy", "S": "I"}, "V", "V"),
			({"root": "shell necklace"}, "N", "DO"),
			({"root": "PRP"}, "PRP", "PRP"),
			({"root": "village"}, "N", "LOC"),
		],
	),
	(
		"veao ukonaghau",
		[
			({"root": "you (S)"}, "PRN", "S"),
			({"root": "see", "S": "you", "DO": "me"}, "V", "V"),
		],
	),
	(
		"Neti ikonari vula noha",
		[
			({"root": "Neti"}, "N", "S"),
			({"root": "see", "S": "he/she", "DO": "them"}, "V", "V"),
			({"root": "shell necklace"}, "N", "DO"),
			({"root": "yesterday"}, "N", "TEMP"),
		],
	),
	(
		"Neti ipasolani tuanga pari kekele",
		[
			({"root": "Neti"}, "N", "S"),
			({"root": "show", "S": "he/she"}, "V", "V"),
			({"root": "PRP-PL"}, "PRP", "PRP"),
			({"root": "child"}, "N", "IO"),
			({"root": "village"}, "N", "DO"),
		],
	),
	(
		"yau ngaaniri niu pa tuanga",
		[
			({"root": "I (S)"}, "PRN", "S"),
			({"root": "eat", "S": "I", "DO": "them"}, "V", "V"),
			({"root": "coconut"}, "N", "DO"),
			({"root": "PRP"}, "PRP", "PRP"),
			({"root": "village"}, "N", "LOC"),
		],
	),
	(
		"tamone tipasolaniri vula pa ghau",
		[
			({"root": "man"}, "N", "S"),
			({"root": "show", "S": "they", "DO": "them"}, "V", "V"),
			({"root": "PRP"}, "PRP", "PRP"),
			({"root": "I (IO)"}, "PRN", "IO"),
			({"root": "shell necklace"}, "N", "DO"),
		],
	),
	(
		"veao upasolani niu pari kaua",
		[
			({"root": "you (S)"}, "PRN", "S"),
			({"root": "show", "S": "you"}, "V", "V"),
			({"root": "PRP-PL"}, "PRP", "PRP"),
			({"root": "dog"}, "N", "IO"),
			({"root": "coconut"}, "N", "DO"),
		],
	),
	(
		"ghaya iani niu lima",
		[
			({"root": "pig"}, "N", "S"),
			({"root": "eat", "S": "he/she"}, "V", "V"),
			({"root": "five"}, "NUM", "NUM"),
			({"root": "coconut"}, "N", "DO"),
		],
	),
	(
		"yau ngaoli kaua pari tamone",
		[
			({"root": "I (S)"}, "PRN", "S"),
			({"root": "buy", "S": "I"}, "V", "V"),
			({"root": "dog"}, "N", "DO"),
			({"root": "PRP-PL"}, "PRP", "PRP"),
			({"root": "man"}, "N", "IO"),
		],
	),
	(
		"tamone tikona vula tolu",
		[
			({"root": "man"}, "N", "S"),
			({"root": "see", "S": "they"}, "V", "V"),
			({"root": "three"}, "NUM", "NUM"),
			({"root": "shell necklace"}, "N", "DO"),
		],
	),
	(
		"veai ikonari tamone pa tuanga",
		[
			({"root": "he/she (S)"}, "PRN", "S"),
			({"root": "see", "S": "he/she", "DO": "them"}, "V", "V"),
			({"root": "man"}, "N", "DO"),
			({"root": "PRP"}, "PRP", "PRP"),
			({"root": "village"}, "N", "LOC"),
		],
	),
]

# UKLO 2022 Advanced P3
zuni = [
	(
		"hoʔa:waptsi",
		[({"root": "chop", "1sg": "1sg", "3pl-abs": "3pl-abs"}, "V", "none")],
	),
	(
		"hoʔaptsi",
		[({"root": "chop", "1sg": "1sg"}, "V", "none")],
	),
	(
		"a:peheʔa",
		[({"root": "tie up", "3pl-abs": "3pl-abs"}, "V", "none")],
	),
	(
		"peheʔanapkä",
		[({"root": "tie up", "tense": "3pl-erg-pst"}, "V", "none")],
	),
	(
		"a:hanlikä",
		[({"root": "steal", "3pl-abs": "3pl-abs", "tense": "pst"}, "V", "none")],
	),
	(
		"hanlikä",
		[({"root": "steal", "tense": "pst"}, "V", "none")],
	),
	(
		"a:wanhatiawa",
		[({"root": "listen", "3pl-abs": "3pl-abs"}, "V", "none")],
	),
	(
		"anhatiawa",
		[({"root": "listen"}, "V", "none")],
	),
	(
		"hoʔa:witcema",
		[({"root": "love", "1sg": "1sg", "3pl-abs": "3pl-abs"}, "V", "none")],
	),
	(
		"uttenapkä",
		[({"root": "bite", "tense": "3pl-erg-pst"}, "V", "none")],
	),
	(
		"a:weʔa",
		[({"root": "be sick", "3pl-abs": "3pl-abs"}, "V", "none")],
	),
	(
		"weʔa",
		[({"root": "be sick"}, "V", "none")],
	),
	(
		"a:pʔɔtuna:we",
		[({"root": "fill", "3pl-abs": "3pl-abs", "tense": "3pl-erg-prs"}, "V", "none")],
	),
	(
		"a:pʔalo",
		[({"root": "bury", "3pl-abs": "3pl-abs"}, "V", "none")],
	),
	(
		"pʔalona:we",
		[({"root": "bury", "tense": "3pl-erg-prs"}, "V", "none")],
	),
	(
		"laʔa",
		[({"root": "grow"}, "V", "none")],
	),
	(
		"pʔiyana:we",
		[({"root": "hang", "tense": "3pl-erg-prs"}, "V", "none")],
	),
	(
		"a:welatenapkä",
		[({"root": "overtake", "tense": "3pl-erg-pst", "3pl-abs": "3pl-abs"}, "V", "none")],
	),
	(
		"elatekä",
		[({"root": "overtake", "tense": "pst"}, "V", "none")],
	),
	(
		"hoʔelate",
		[({"root": "overtake", "1sg": "1sg"}, "V", "none")],
	),
	(
		"a:pʔalokä",
		[({"root": "bury", "3pl-abs": "3pl-abs", "tense": "pst"}, "V", "none")],
	),
	(
		"hanlina:we",
		[({"root": "steal", "tense": "3pl-erg-prs"}, "V", "none")],
	),
]


### SCRAMBLED ROSETTA ###

# NACLO 2018 R2, problem P
beja = (
	[
		"Tak rihan",
		"Yaas rihan",
		"Akra tak rihan",
		"Dabalo yaas rihan",
		"Tak akraab rihan",
		"Tak dabaloob rihan",
		"Tak akteen",
		"Rihane tak akteen",
		"Tak rihaneeb akteen",
	],
	[
		[
			({"root": "see"}, "V", "V"),
			({"root": "man"}, "N", "O"),
			({"root": "strong-REL"}, "REL", "REL"),
		],
		[
			({"root": "know"}, "V", "V"),
			({"root": "man"}, "N", "O"),
			({"root": "seen-REL"}, "REL", "REL"),
		],
		[
			({"root": "know"}, "V", "V"),
			({"root": "man"}, "N", "O"),
			({"root": "seen-ADJ"}, "ADJ", "ADJ"),
		],
		[
			({"root": "see"}, "V", "V"),
			({"root": "man"}, "N", "O"),
			({"root": "small-REL"}, "REL", "REL"),
		],
		[
			({"root": "see"}, "V", "V"),
			({"root": "dog"}, "N", "O"),
			({"root": "small-ADJ"}, "ADJ", "ADJ"),
		],
		[
			({"root": "see"}, "V", "V"),
			({"root": "man"}, "N", "O"),
			({"root": "strong-ADJ"}, "ADJ", "ADJ"),
		],
		[
			({"root": "see"}, "V", "V"),
			({"root": "dog"}, "N", "O"),
		],
		[
			({"root": "see"}, "V", "V"),
			({"root": "man"}, "N", "O"),
		],
		[
			({"root": "know"}, "V", "V"),
			({"root": "man"}, "N", "O"),
		],
	],
)


### SEMANTIC MATCHING ###

# IOL 2005 P2
lango = [
	[
		"dyè ɔ̀t",
		"dyè tyɛ̀n",
		"gìn",
		"gìn wìc",
		"ɲíg",
		"ɲíg wàŋ",
		"ɔ̀t cɛ̀m",
		"wìc ɔ̀t",
	],
	[
		[
			({"root": "grain"}, "N", "possessed"),
			({"root": "eye"}, "N", "possessor"),
		],
		[
			({"root": "grain"}, "N", "possessed"),
		],
		[
			({"root": "head"}, "N", "possessed"),
			({"root": "house"}, "N", "possessor"),
		],
		[
			({"root": "garment"}, "N", "possessed"),
		],
		[
			({"root": "bottom"}, "N", "possessed"),
			({"root": "house"}, "N", "possessor"),
		],
		[
			({"root": "house"}, "N", "possessed"),
			({"root": "food"}, "N", "possessor"),
		],
		[
			({"root": "bottom"}, "N", "possessed"),
			({"root": "foot"}, "N", "possessor"),
		],
		[
			({"root": "garment"}, "N", "possessed"),
			({"root": "head"}, "N", "possessor"),
		],
	],
]

# Ukraine LO 2018/19, TST tiebreaker, P3
sanskrit = [
	[
		"yah",
		"tatɦā",
		"sarvatra",
		"ekah",
		"yadā",
		"tatra",
		"yatra",
		"sarvah",
	],
	[
		[({"type": "all", "root": "where"}, "PRN", "none")],
		[({"type": "interrogative", "root": "where"}, "PRN", "none")],
		[({"type": "all", "root": "which"}, "PRN", "none")],
		[({"type": "interrogative", "root": "when"}, "PRN", "none")],
		[({"type": "interrogative", "root": "which"}, "PRN", "none")],
		[({"type": "demonstrative", "root": "how"}, "PRN", "none")],
		[({"type": "demonstrative", "root": "where"}, "PRN", "none")],
		[({"type": "same", "root": "which"}, "PRN", "none")],
	]
]

# IOL 2016 P4
iatmul = [
	[
		"guna vaala",
		"kaʔik",
		"kaʔikgu",
		"klawun",
		"laavu",
		"laavuga viʔ",
		"laavuga",
		"niʔbu",
		"niʔbuna vaala",
		"nyakaʔik",
		"vi",
		"viʔwun",
		"waliniʔbana bâk",
		"waliniʔbana gu",
		"waliniʔbana vi",
	],
	[
		[
			({"root": "banana"}, "N", "N"),
		],
		[
			({"root": "white people", "suffix": "possessor"}, "N", "possessor"),
			({"root": "pig"}, "N", "N"),
		],
		[
			({"root": "water", "suffix": "possessor"}, "N", "possessor"),
			({"root": "canoe"}, "N", "N"),
		],
		[
			({"root": "banana", "suffix": "leaves"}, "N", "N"),
		],
		[
			({"root": "white people", "suffix": "possessor"}, "N", "possessor"),
			({"root": "water"}, "N", "N"),
		],
		[
			({"root": "white people", "suffix": "possessor"}, "N", "possessor"),
			({"root": "spear"}, "N", "N"),
		],
		[
			({"root": "picture/shadow"}, "N", "N"),
		],
		[
			({"root": "ground/land", "suffix": "possessor"}, "N", "possessor"),
			({"root": "canoe"}, "N", "N"),
		],
		[
			({"root": "to get", "suffix": "1sg.pst"}, "V", "V"),
		],
		[
			({"root": "picture/shadow", "suffix": "water"}, "N", "N"),
		],
		[
			({"root": "to see", "suffix": "1sg.pst"}, "V", "V"),
		],
		[
			({"root": "sun", "suffix": "picture/shadow"}, "N", "N"),
		],
		[
			({"root": "spear"}, "N", "N"),
		],
		[
			({"root": "banana", "suffix": "leaves"}, "N", "N"),
			({"root": "to see"}, "V", "V"),
		],
		[
			({"root": "ground/land"}, "N", "N"),
		],
	],
]

# TOL 2026 R2 P3
haida = [
	[
		"Gáan skáa sGwáansang",
		"Gáan sk’a hlGúnahl",
		"Gáadaangw Ga hlGúnahl",
		"kit’uu sk’a sGwáansang",
		"kit’uu dáagal sGa stánsang",
		"kíihlaa dláanwaay Ga sGwáansang",
		"kíihlaa tl’a sGwáansang",
		"k’án sGa sGwáansang",
		"k’án sk’a sdáng",
		"sándiigaa sGa tléehl",
		"sgi skáajaaw skáa stánsang",
		"sgíndaaw tl’a hlGúnahl",
		"skáy Ga stánsang",
		"skáy skáa hlGúnahl",
		"stáw stlíin sk’a tléehl",
		"xyáay sk’a sdáng",
		"xyáay tl’a stánsang",
		"xángaang dláanwaay Ga stánsang",
	],
	[
		[
			({"root": "one"}, "NUM", "NUM"),
			({"root": "flat"}, "Q", "Q"),
			({"root": "tray"}, "N", "N"),
		],
		[
			({"root": "one"}, "NUM", "NUM"),
			({"root": "container"}, "Q", "Q"),
			({"root": "basin"}, "N", "N"),
			({"root": "tray"}, "N", "MOD"),
		],
		[
			({"root": "one"}, "NUM", "NUM"),
			({"root": "spherical"}, "Q", "Q"),
			({"root": "berry"}, "N", "N"),
		],
		[
			({"root": "one"}, "NUM", "NUM"),
			({"root": "slender.hard"}, "Q", "Q"),
			({"root": "harpoon"}, "N", "N"),
		],
		[
			({"root": "one"}, "NUM", "NUM"),
			({"root": "other"}, "Q", "Q"),
			({"root": "grass"}, "N", "N"),
		],
		[
			({"root": "two"}, "NUM", "NUM"),
			({"root": "slender.hard"}, "Q", "Q"),
			({"root": "arm"}, "N", "N"),
		],
		[
			({"root": "two"}, "NUM", "NUM"),
			({"root": "slender.hard"}, "Q", "Q"),
			({"root": "grass"}, "N", "N"),
		],
		[
			({"root": "three"}, "NUM", "NUM"),
			({"root": "flat"}, "Q", "Q"),
			({"root": "rudder"}, "N", "N"),
		],
		[
			({"root": "three"}, "NUM", "NUM"),
			({"root": "spherical"}, "Q", "Q"),
			({"root": "shell"}, "N", "N"),
		],
		[
			({"root": "three"}, "NUM", "NUM"),
			({"root": "container"}, "Q", "Q"),
			({"root": "bathtub"}, "N", "N"),
		],
		[
			({"root": "three"}, "NUM", "NUM"),
			({"root": "slender.hard"}, "Q", "Q"),
			({"root": "berry"}, "N", "N"),
		],
		[
			({"root": "four"}, "NUM", "NUM"),
			({"root": "flat"}, "Q", "Q"),
			({"root": "arm"}, "N", "N"),
		],
		[
			({"root": "four"}, "NUM", "NUM"),
			({"root": "other"}, "Q", "Q"),
			({"root": "rope"}, "N", "N"),
			({"root": "harpoon"}, "N", "MOD"),
		],
		[
			({"root": "four"}, "NUM", "NUM"),
			({"root": "container"}, "Q", "Q"),
			({"root": "shell"}, "N", "N"),
		],
		[
			({"root": "four"}, "NUM", "NUM"),
			({"root": "container"}, "Q", "Q"),
			({"root": "basin"}, "N", "N"),
			({"root": "washing"}, "N", "MOD"),
		],
		[
			({"root": "four"}, "NUM", "NUM"),
			({"root": "spherical"}, "Q", "Q"),
			({"root": "ball"}, "N", "N"),
			({"root": "baseball"}, "N", "MOD"),
		],
		[
			({"root": "five"}, "NUM", "NUM"),
			({"root": "slender.hard"}, "Q", "Q"),
			({"root": "spine"}, "N", "N"),
			({"root": "sea urchin"}, "N", "MOD"),
		],
		[
			({"root": "five"}, "NUM", "NUM"),
			({"root": "other"}, "Q", "Q"),
			({"root": "week"}, "N", "N"),
		],
	],
]


if __name__ == "__main__":
	solve(mizo, type="rosetta")
	solve(apinaye, type="rosetta")
	solve(nung, type="rosetta")
	solve(arrernte, type="rosetta")
	solve(amele, type="rosetta")
	solve(zapotec, type="rosetta")
	solve(kove, type="rosetta")
	solve(zuni, type="rosetta")
	solve(beja, type="scrambled_rosetta")
	solve(lango, type="scrambled_rosetta")
	solve(sanskrit, type="scrambled_rosetta")
	solve(iatmul, type="scrambled_rosetta")
	solve(haida, type="scrambled_rosetta")