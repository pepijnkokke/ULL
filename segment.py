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


class Corpus:

    def numWords(self):
        """ Compute the number of words in the corpus. """
        # NOTE: I have the expectation that this will be more or less O(1)
        pass

    def countWord(self,word):
        """ Compute the number of times that a lexical item occurs. """
        # NOTE: I have the expectation that this will be more or less O(1)
        pass

    def pPhoneme(self,phon):
        """ Compute the prior probability of a phoneme. """
        pass


Param = namedtuple('Param', ['alpha', 'pEndWord', 'pEndUtt', 'corpus'])


class Dist(Param):
    """ Implementation of the base distribution """

    def prob(self,word):
        """ Compute the probability of generating a given word. """

        chanceOfNovel =
            self.param.alpha / float(self.corpus.numWords() + self.param.alpha)

        if random() < chanceOfNovel:

            # P(generating M phonemes, followed by a bound)
            p1 = (1 - self.param.pEndWord) ** (len(word) - 1) * sefl.param.pEndWord

            # P(generating the given sequence of phonemes)
            p2 = reduce(mul, map(self.corpus.pPhoneme,word))

            return p1 * p2

        else:

            return self.corpus.countWord(word) / float(self.corpus.numWords())

def evaluate(segmented_found,segmented_true,lexicon_found,lexicon_true):
    """ Evaluate precision, recall and F0 for:
        - Words (word boundaries placed correctly before and after the word)
        - Lexical (lexical types found)
        - "Potentially ambiguous boundaries, i.e. utterance boundaries not included in counts". Kind of vague, will ask on thursday
    """
    #Note: these are exactly the evaluations as done by Goldwater & friends
    #words_found = #something with segmented_found
    #words_true = #something with segmented_true
    #P = precision(words_found,words_true)
    #R = recall(words_found,words_true)
    #F = f_zero(P,R)
    LP = precision(lexicon_found,lexicon_true)
    LR = recall(lexicon_found,lexicon_true)
    LF = f_zero(LP/LR)
    #BP =
    #BR =
    #BF = f_zero(BP,BR)
def precision(found,true):
    """ Number of correct / number found """
    #ToDo
    
def recall(found,true):
    """ Number of correct / number of all correct """
    #ToDo
    
def f_zero(precision,recall):
    """ Geometric average of precision and recall """
    return (2*precision*recall)/(precision+recall)

def find_enclosing_boundaries(boundaries, i):
    """ Find the nearest boundaries on both sides of i """
    lower = max([x for x in boundaries if x < i])
    upper = min([x for x in boundaries if x > i])

    return lower, upper


def words_in_utterance(utterance, boundaries):
    """ Convert an utterance to a list of words based on the given boundaries """
    out = ''
    for i, phoneme in enumerate(utterance):
        if i in boundaries:
            out += ' '

        out += phoneme

    return out.split(' ')


def gibbs_iteration(corpus, dist):
    for (utterance, boundaries) in corpus:
        for i, phoneme in enumerate(utterance):
            lower, upper = find_enclosing_boundaries(boundaries, i)
            w1 = utterance[lower:upper]
            w2 = utterance[lower:i]
            w3 = utterance[i:upper]

            words = words_in_utterance(utterance, boundaries)

            # subtract 1 from the counts to compensate for counting itself
            n = len(words) - 1
            p_h1 = ((words.count(w1) - 1 + dist.alpha * dist.prob(w1)) /
                    (n + dist.alpha)) #TODO: times the utterance-final thing

            p_h2 = ((words.count(w2) + dist.alpha * dist.prob(w2)) / (n + dist.alpha) *
                    # TODO: utterance-final thing
                    (words.count(w3) + 1 if w2 == w3 else 0 + dist.alpha *
                     dist.prob(w3)) / (n + 1 + dist.alpha))
                    # TODO: utterance-final thing
