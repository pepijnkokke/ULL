from operator import mul
from random   import random


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


def numberOfWords(corpus):
    """
    Compute the number of words in the corpus.
    """
    pass



def frequencyOfWord(word,corpus):
    """
    Compute the number of times that a lexical item occurs.
    """
    pass



def p(alpha,pBound,word,corpus):
    """
    Compute the probability of generating a given word.
    """

    num           = numberOfWords(corpus)
    chanceOfNovel = alpha / (num + alpha)

    if random() < chanceOfNovel:

        # P(generating M phonemes, followed by a bound)
        pBounds   = (1 - pBound) ** (len(word) - 1) * pBound

        # P(generating the given sequence of phonemes)
        pPhonemes = reduce(mul, map(pPhoneme,word))
        return pBound * pPhonemes

    else:

        freq = frequencyOfWord(word,corpus)
        return freq / float(num)


def pPhoneme():
    """
    Compute the probability of generating a given phoneme.
    """
    pass


def chineseRestaurantProcess(words):
    """
    Determine the table at which the ith guest sits down, using a Chinese
    restaurant process.
    """
    pass
