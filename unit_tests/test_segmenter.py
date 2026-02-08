import sys
from pathlib import Path

_bpl_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_bpl_dir))

from edit_distance_calculator import EditDistanceCalculator
from features import FeatureBank
from segmenter import Segmenter


def root_of(feats):
    for slot, val in feats:
        if slot == "root":
            return val


def test_permyak():
    words = [
        ("cerkulaɲ", (("root", "house"), ("case", "towards"))),
        ("pɨzannezɨtlən", (("root", "desk"), ("case", "of"), ("number", "pl"), ("poss", "your-sg"))),
        ("ponɨt", (("root", "dog"), ("poss", "your-sg"))),
        ("purtnɨs", (("root", "knife"), ("poss", "their"))),
        ("kəinnezɨs", (("root", "wolf"), ("number", "pl"), ("poss", "his"))),
        ("vərələn", (("root", "forest"), ("case", "of"), ("poss", "my"))),
        ("purtəla", (("root", "knife"), ("case", "for the sake of"), ("poss", "my"))),
        ("tɨezɨtkət", (("root", "lake"), ("case", "with"), ("number", "pl"), ("poss", "your-sg"))),
        ("cerkuezliɕ", (("root", "house"), ("case", "from"), ("number", "pl"))),
        ("juɕɕezə", (("root", "swan"), ("number", "pl"), ("poss", "my"))),
        ("kokɨskət", (("root", "foot"), ("case", "with"), ("poss", "his"))),
        ("ciɨtlaɲ", (("root", "hand"), ("case", "towards"), ("poss", "your-sg"))),
        ("pɨzanɨsliɕ", (("root", "desk"), ("case", "from"), ("poss", "his"))),
        ("vərrezlən", (("root", "forest"), ("case", "of"), ("number", "pl"))),
        ("ponnɨt", (("root", "dog"), ("poss", "your-pl"))),
        ("juɕla", (("root", "swan"), ("case", "for the sake of"))),
    ]
    words = sorted(words, key=lambda wf: (root_of(wf[1]), wf[1]))
    word_to_present_slots = {
        (word, feats): {slot for (slot, _) in feats} for (word, feats) in words
    }
    word_to_slot_values = {
        (word, feats): {slot: value for (slot, value) in feats} for (word, feats) in words
    }
    ed = EditDistanceCalculator(FeatureBank([w for (w, _) in words]))
    segmenter = Segmenter(
		allow_empty_slots=True,
		segmentations_beyond_minimal=0,
		segmentation_queue_limit=100000,
		segmentation_queue_trim_factor=1.5,
		words=words,
		word_to_present_slots=word_to_present_slots,
		word_to_slot_values=word_to_slot_values,
		ed=ed,
		debug=False,
    )
    segmentations = segmenter.generate_segmentations(order=["root", "number", "poss", "case"])
    print(segmentations, flush=True)
    expected = [
        {
            ('pɨzanɨsliɕ', (('root', 'desk'), ('case', 'from'), ('poss', 'his'))): [0, 5, 5, 7],
            ('pɨzannezɨtlən', (('root', 'desk'), ('case', 'of'), ('number', 'pl'), ('poss', 'your-sg'))): [0, 5, 8, 10],
            ('ponnɨt', (('root', 'dog'), ('poss', 'your-pl'))): [0, 3, 3, 6],
            ('ponɨt', (('root', 'dog'), ('poss', 'your-sg'))): [0, 3, 3, 5],
            ('kokɨskət', (('root', 'foot'), ('case', 'with'), ('poss', 'his'))): [0, 3, 3, 5],
            ('vərrezlən', (('root', 'forest'), ('case', 'of'), ('number', 'pl'))): [0, 3, 6, 6],
            ('vərələn', (('root', 'forest'), ('case', 'of'), ('poss', 'my'))): [0, 3, 3, 4],
            ('ciɨtlaɲ', (('root', 'hand'), ('case', 'towards'), ('poss', 'your-sg'))): [0, 2, 2, 4],
            ('cerkuezliɕ', (('root', 'house'), ('case', 'from'), ('number', 'pl'))): [0, 5, 7, 7],
            ('cerkulaɲ', (('root', 'house'), ('case', 'towards'))): [0, 5, 5, 5],
            ('purtəla', (('root', 'knife'), ('case', 'for the sake of'), ('poss', 'my'))): [0, 4, 4, 5],
            ('purtnɨs', (('root', 'knife'), ('poss', 'their'))): [0, 4, 4, 7],
            ('tɨezɨtkət', (('root', 'lake'), ('case', 'with'), ('number', 'pl'), ('poss', 'your-sg'))): [0, 2, 4, 6],
            ('juɕla', (('root', 'swan'), ('case', 'for the sake of'))): [0, 3, 3, 3],
            ('juɕɕezə', (('root', 'swan'), ('number', 'pl'), ('poss', 'my'))): [0, 3, 6, 7],
            ('kəinnezɨs', (('root', 'wolf'), ('number', 'pl'), ('poss', 'his'))): [0, 4, 7, 9]
        },
        {
            ('pɨzanɨsliɕ', (('root', 'desk'), ('case', 'from'), ('poss', 'his'))): [0, 5, 5, 7],
            ('pɨzannezɨtlən', (('root', 'desk'), ('case', 'of'), ('number', 'pl'), ('poss', 'your-sg'))): [0, 6, 8, 10],
            ('ponnɨt', (('root', 'dog'), ('poss', 'your-pl'))): [0, 3, 3, 6],
            ('ponɨt', (('root', 'dog'), ('poss', 'your-sg'))): [0, 3, 3, 5],
            ('kokɨskət', (('root', 'foot'), ('case', 'with'), ('poss', 'his'))): [0, 3, 3, 5],
            ('vərrezlən', (('root', 'forest'), ('case', 'of'), ('number', 'pl'))): [0, 4, 6, 6],
            ('vərələn', (('root', 'forest'), ('case', 'of'), ('poss', 'my'))): [0, 3, 3, 4],
            ('ciɨtlaɲ', (('root', 'hand'), ('case', 'towards'), ('poss', 'your-sg'))): [0, 2, 2, 4],
            ('cerkuezliɕ', (('root', 'house'), ('case', 'from'), ('number', 'pl'))): [0, 5, 7, 7],
            ('cerkulaɲ', (('root', 'house'), ('case', 'towards'))): [0, 5, 5, 5],
            ('purtəla', (('root', 'knife'), ('case', 'for the sake of'), ('poss', 'my'))): [0, 4, 4, 5],
            ('purtnɨs', (('root', 'knife'), ('poss', 'their'))): [0, 4, 4, 7],
            ('tɨezɨtkət', (('root', 'lake'), ('case', 'with'), ('number', 'pl'), ('poss', 'your-sg'))): [0, 2, 4, 6],
            ('juɕla', (('root', 'swan'), ('case', 'for the sake of'))): [0, 3, 3, 3],
            ('juɕɕezə', (('root', 'swan'), ('number', 'pl'), ('poss', 'my'))): [0, 4, 6, 7],
            ('kəinnezɨs', (('root', 'wolf'), ('number', 'pl'), ('poss', 'his'))): [0, 5, 7, 9]
        },
    ]
    assert any(seg_dict in expected for (seg_dict, _) in segmentations)


def test_zoque():
    words = [
        ("nakpatpit", (("root", "cactus"), ("case", "with"), ("poss", "poss"))),
        ("nakpat", (("root", "cactus"),)),
        ("mokpittih", (("root", "corn"), ("case", "with"), ("only", "only"))),
        ("pokskukyʌsmʌtaʔm", (("root", "chair"), ("case", "above"), ("number", "pl"))),
        ("pokskuy", (("root", "chair"),)),
        ("peroltih", (("root", "kettle"), ("only", "only"))),
        ("kot^sʌktaʔm", (("root", "mountain"), ("number", "pl"))),
        ("komgʌsmʌtih", (("root", "post"), ("case", "above"), ("only", "only"))),
        ("ŋgom", (("root", "post"), ("poss", "poss"))),
        ("kʌmʌŋbitšeh", (("root", "shadow"), ("case", "with"), ("like", "like"))),
        ("kʌmʌŋdaʔm", (("root", "shadow"), ("number", "pl"))),
        ("nd^zapkʌsmʌšeh", (("root", "sky"), ("case", "above"), ("like", "like"), ("poss", "poss"))),
        ("t^sapšeh", (("root", "sky"), ("like", "like"))),
        ("pahsungotoya", (("root", "squash"), ("case", "for"))),
        ("pahsunšehtaʔmdih", (("root", "squash"), ("like", "like"), ("number", "pl"), ("only", "only"))),
        ("tʌt^skotoyatih", (("root", "tooth"), ("case", "for"), ("only", "only"))),
        ("kumgukyʌsmʌ", (("root", "town"), ("case", "above"))),
        ("kumgukyotoyataʔm", (("root", "town"), ("case", "for"), ("number", "pl"))),
        ("t^sakyotoya", (("root", "vine"), ("case", "for"))),
        ("nd^zay", (("root", "vine"), ("poss", "poss"))),
        ("t^sakyʌsmʌtih", (("root", "vine"), ("case", "above"), ("only", "only"))),
        ("kʌmʌŋšeh", (("root", "shadow"), ("like", "like"))),
        ("mok", (("root", "corn"), ("poss", "poss"))),
        ("ndʌt^staʔm", (("root", "tooth"), ("number", "pl"), ("poss", "poss"))),
        ("pahsunbit", (("root", "squash"), ("case", "with"))),
        ("perolkotoyašehtaʔm", (("root", "kettle"), ("case", "for"), ("like", "like"), ("number", "pl"))),
    ]
    words = sorted(words, key=lambda wf: (root_of(wf[1]), wf[1]))
    word_to_present_slots = {
        (word, feats): {slot for (slot, _) in feats} for (word, feats) in words
    }
    word_to_slot_values = {
        (word, feats): {slot: value for (slot, value) in feats} for (word, feats) in words
    }
    ed = EditDistanceCalculator(FeatureBank([w for (w, _) in words]))
    segmenter = Segmenter(
		allow_empty_slots=True,
		segmentations_beyond_minimal=0,
		segmentation_queue_limit=100000,
		segmentation_queue_trim_factor=1.5,
		words=words,
		word_to_present_slots=word_to_present_slots,
		word_to_slot_values=word_to_slot_values,
		ed=ed,
		debug=False,
    )
    segmentations = segmenter.generate_segmentations(order=['poss', 'root', 'case', 'like', 'number', 'only'])
    print(segmentations, flush=True)
    expected = {
        ('nakpat', (('root', 'cactus'),)): [0, 0, 6, 6, 6, 6],
        ('nakpatpit', (('root', 'cactus'), ('case', 'with'), ('poss', 'poss'))): [0, 0, 6, 9, 9, 9],
        ('pokskuy', (('root', 'chair'),)): [0, 0, 7, 7, 7, 7],
        ('pokskukyʌsmʌtaʔm', (('root', 'chair'), ('case', 'above'), ('number', 'pl'))): [0, 0, 7, 12, 12, 16],
        ('mokpittih', (('root', 'corn'), ('case', 'with'), ('only', 'only'))): [0, 0, 3, 6, 6, 6],
        ('mok', (('root', 'corn'), ('poss', 'poss'))): [0, 0, 3, 3, 3, 3],
        ('perolkotoyašehtaʔm', (('root', 'kettle'), ('case', 'for'), ('like', 'like'), ('number', 'pl'))): [0, 0, 5, 11, 14, 18],
        ('peroltih', (('root', 'kettle'), ('only', 'only'))): [0, 0, 5, 5, 5, 5],
        ('kot^sʌktaʔm', (('root', 'mountain'), ('number', 'pl'))): [0, 0, 7, 7, 7, 11],
        ('komgʌsmʌtih', (('root', 'post'), ('case', 'above'), ('only', 'only'))): [0, 0, 3, 8, 8, 8],
        ('ŋgom', (('root', 'post'), ('poss', 'poss'))): [0, 1, 4, 4, 4, 4],
        ('kʌmʌŋbitšeh', (('root', 'shadow'), ('case', 'with'), ('like', 'like'))): [0, 0, 5, 8, 11, 11],
        ('kʌmʌŋšeh', (('root', 'shadow'), ('like', 'like'))): [0, 0, 5, 5, 8, 8],
        ('kʌmʌŋdaʔm', (('root', 'shadow'), ('number', 'pl'))): [0, 0, 5, 5, 5, 9],
        ('nd^zapkʌsmʌšeh', (('root', 'sky'), ('case', 'above'), ('like', 'like'), ('poss', 'poss'))): [0, 1, 6, 11, 14, 14],
        ('t^sapšeh', (('root', 'sky'), ('like', 'like'))): [0, 0, 5, 5, 8, 8],
        ('pahsungotoya', (('root', 'squash'), ('case', 'for'))): [0, 0, 6, 12, 12, 12],
        ('pahsunbit', (('root', 'squash'), ('case', 'with'))): [0, 0, 6, 9, 9, 9],
        ('pahsunšehtaʔmdih', (('root', 'squash'), ('like', 'like'), ('number', 'pl'), ('only', 'only'))): [0, 0, 6, 6, 9, 13],
        ('tʌt^skotoyatih', (('root', 'tooth'), ('case', 'for'), ('only', 'only'))): [0, 0, 5, 11, 11, 11],
        ('ndʌt^staʔm', (('root', 'tooth'), ('number', 'pl'), ('poss', 'poss'))): [0, 1, 6, 6, 6, 10],
        ('kumgukyʌsmʌ', (('root', 'town'), ('case', 'above'))): [0, 0, 6, 11, 11, 11],
        ('kumgukyotoyataʔm', (('root', 'town'), ('case', 'for'), ('number', 'pl'))): [0, 0, 6, 12, 12, 16],
        ('t^sakyʌsmʌtih', (('root', 'vine'), ('case', 'above'), ('only', 'only'))): [0, 0, 5, 10, 10, 10],
        ('t^sakyotoya', (('root', 'vine'), ('case', 'for'))): [0, 0, 5, 11, 11, 11],
        ('nd^zay', (('root', 'vine'), ('poss', 'poss'))): [0, 1, 6, 6, 6, 6]}
    assert expected in [seg_dict for (seg_dict, _) in segmentations]


if __name__ == "__main__":
    test_permyak()
    test_zoque()
