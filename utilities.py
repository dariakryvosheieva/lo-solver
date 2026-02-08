import heapq, os, random, sys, tempfile, re


def _normalize(token):
	"""Lowercase; remove leading/trailing whitespace and punctuation."""
	t = token.strip().lower()
	return re.sub(r"^[^\w]+|[^\w]+$", "", t, flags=re.UNICODE)


def parse(sentence):
	"""Convert sentence into a list of word tokens."""
	return [w for w in (_normalize(t) for t in sentence.split()) if w]


def feature_dict_to_tuple(features):
	"""
	Convert an input feature dict into a tuple of (key, value) pairs.
	The tuple is sorted so that root is first, followed by other features sorted lexicographically.
	Example: {'case': 'nom', 'number': 'pl', 'root': 'problem'} -> 
					(('root', 'problem'), ('case', 'nom'), ('number', 'pl'))
	"""
	return tuple(sorted(features.items(), key=lambda x: (x[0] != "root", x[0])))


def flushEverything():
    sys.stdout.flush()
    sys.stderr.flush()


def makeTemporaryFile(suffix, d="."):
    if d != "." and not os.path.exists(d):
        os.makedirs(d)
    fd = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, dir=d)
    fd.write("")
    fd.close()
    return fd.name


VERBOSITYLEVEL = 0


def getVerbosity():
    global VERBOSITYLEVEL
    return VERBOSITYLEVEL


def sampleGeometric(p):
    if random.random() < p:
        return 0
    return 1 + sampleGeometric(p)


class PQ:
    def __init__(self):
        self.h = []

    def push(self, priority, v):
        heapq.heappush(self.h, (-priority, v))

    def popMaximum(self):
        return heapq.heappop(self.h)[1]

    def __iter__(self):
        for _, v in self.h:
            yield v

    def __len__(self):
        return len(self.h)


def randomlyRemoveOne(xs):
    j = random.choice(list(range(len(xs))))
    return xs[:j] + xs[j + 1 :]


class RunWithTimeout(Exception):
    pass
