# import torch
from nltk.util import ngrams
from collections import Counter
import configuration as config
import const

#ngram extraction
def getCommonGrams(words):
    bigrams = {}
    trigrams = {}
    for word in words:
        for n in [2, 3]:
            currgrams = ngrams(word, n)
            for gram in currgrams:
                gram = ''.join(gram)
                if n == 2:
                    bigrams[gram] = bigrams[gram] + 1 if gram in bigrams else 1
                else:
                    trigrams[gram] = trigrams[gram] + 1 if gram in trigrams else 1

    const.bigrams = [x.pop(0) for x in [list(y) for y in (Counter(bigrams).most_common(config.di_num))]]
    const.trigrams =[x.pop(0) for x in [list(y) for y in (Counter(trigrams).most_common(config.tri_num))]]

def get_suffixes(words):
    letters = {}
    difixes = {}
    trifixes = {}

    for word in words:
        letters[word[-1]] = letters[word[-1]] + 1 if word[-1] in letters else 1
        difixes[word[-2:]] = difixes[word[-2:]] + 1 if word[-2:] in difixes else 1
        trifixes[word[-3:]] = trifixes[word[-3:]] + 1 if word[-3:] in trifixes else 1
    const.last_letters = [x.pop(0) for x in [list(y) for y in (Counter(letters).most_common(config.last_letters))]]
    const.di_suffix = [x.pop(0) for x in [list(y) for y in (Counter(difixes).most_common(config.di_sufnum))]]
    const.tri_suffix =[x.pop(0) for x in [list(y) for y in (Counter(trifixes).most_common(config.tri_sufnum))]]
