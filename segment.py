from collections import namedtuple
from operator    import mul
from random      import random


# w         : input word
# l         : lexical item
# n         : number of previously generated words
# n^l       : number of times lexical item l has occured in the n words
# alpha     : parameter of the model
# P#        : probability of generating a word boundary
# phonemes  : set of phonemes
# cache     : in principle all previously segmented utterances, but in
#             practice we only have to store n and n^l
# utterance : unsegmented string of phonemes
# words     : segmented utterance

# corpus : [[[Phoneme]]] (TODO: replace corpus with a dict which tracks n and n^l)


Param  = namedtuple('Param', ['alpha', 'pEndWord', 'pEndUtt'])


class Corpus:

    def numWords(self):
        """ Compute the number of words in the corpus. """
        pass

    def countWord(self,word):
        """ Compute the number of times that a lexical item occurs. """
        pass

    def pPhoneme(self,phon):
        """ Computer the prior probability of a phoneme. """
        pass


def prob(param,word,corpus):
    """
    Compute the probability of generating a given word.
    """

    chanceOfNovel = param.alpha / float(corpus.numWords + param.alpha)

    if random() < chanceOfNovel:

        # P(generating M phonemes, followed by a bound)
        p1 = (1 - param.pEndWord) ** (len(word) - 1) * param.pEndWord

        # P(generating the given sequence of phonemes)
        p2 = reduce(mul, map(corpus.pPhoneme,word))

        return p1 * p2

    else:

        return corpus.countWord(word) / float(corpus.numWords)


def chineseRestaurantProcess(words):
    """
    Determine the table at which the ith guest sits down, using a Chinese
    restaurant process.
    """
    pass
