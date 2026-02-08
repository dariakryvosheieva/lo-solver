from features import tokenize


class Morph:
    def __init__(self, phonemes):
        if isinstance(phonemes, str):
            phonemes = tokenize(phonemes)
        self.phonemes = phonemes

    def __unicode__(self):
        return "/{}/".format("".join(self.phonemes))

    def __str__(self):
        return self.__unicode__()

    def __repr__(self):
        return str(self)

    # this interferes with parallel computation - probably because it messes up serialization
    # def __repr__(self): return unicode(self)
    def __len__(self):
        return len(self.phonemes)

    def __add__(self, other):
        return Morph(self.phonemes + other.phonemes)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(tuple(self.phonemes))

    def __ne__(self, other):
        return str(self) != str(other)

    def __getitem__(self, sl):
        if isinstance(sl, int):
            return self.phonemes[sl]
        if isinstance(sl, slice):
            return Morph(self.phonemes[sl.start : sl.stop : sl.step])
