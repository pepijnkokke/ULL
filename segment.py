from sys         import argv
from collections import namedtuple
from operator    import mul
from random      import random
from copy        import copy


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
    def __init__(self, path):
        self.utt_boundaries = [0]
        self.text = ''

        with open(path, 'r') as f:
            for utterance in f:
                self.text += utterance
                self.utt_boundaries.append(self.utt_boundaries[-1] +
                                           len(utterance))

        self.boundaries = copy(self.utt_boundaries)

    def numWords(self):
        """ Compute the number of words in the corpus. """
        # NOTE: I have the expectation that this will be more or less O(1)
        pass

    def countWord(self, word):
        """ Compute the number of times that a lexical item occurs. """
        # NOTE: I have the expectation that this will be more or less O(1)
        pass

    def pPhoneme(self, phon):
        """ Compute the prior probability of a phoneme. """
        return float(self.text.count(phon)) / len(self.text)

    def p0(self, word, p_hashtag=0.5):
        p = reduce(mul, map(self.pPhoneme, word))
        return p_hashtag * (1 - p_hashtag)**(len(word) - 1) * p




def evaluate(segmented_found,segmented_true,lexicon_found,lexicon_true):
    """ Evaluate precision, recall and F0 for:
        - Words (word boundaries placed correctly before and after the word)
        - Lexical (lexical types found)
        - Boundaries (rather than looking at words, look at just the boundary positions excluding the utterance boundaries, as they are already correct)
    """
    #Note: these are exactly the evaluations as done by Goldwater & friends
    #words_found = #something with segmented_found
    #words_true = #something with segmented_true
    #P,R = precision_recall(words_found,words_true)
    #F = f_zero(P,R)
    LP,LR = precision_recall(lexicon_found,lexicon_true)
    LF = f_zero(LP/LR)
    #BP,BR =
    #BF = f_zero(BP,BR)

def precision_recall(found_items,true_items):
    """ Number of correct / number found """
    c = 0
    #For each word in found_items
    for item in found_items:

        #If exists in true_items
        if item in true_items:
            c += 1
            
    return (c/len(found_items), c/len(true_items))

def f_zero(precision,recall):
    """ Geometric average of precision and recall """
    return (2*precision*recall)/(precision+recall)

def find_enclosing_boundaries(boundaries, i):
    """ Find the nearest boundaries on both sides of i """
    lower = max([x for x in boundaries if x < i])
    upper = min([x for x in boundaries if x > i])

    return lower, upper


def split_on_boundaries(text, boundaries):
    """ Convert an utterance to a list of words based on the given boundaries """
    out = ''
    for i, phoneme in enumerate(text):
        if i in boundaries:
            out += ' '

        out += phoneme

    return out.split(' ')


def gibbs_iteration(corpus, rho=2.0, alpha=0.5):
    words = split_on_boundaries(corpus.text, corpus.boundaries)
    n = len(words) - 1

    for i, phoneme in enumerate(corpus.text):
        print i
        if i in corpus.utt_boundaries:
            continue

        lower, upper = find_enclosing_boundaries(corpus.boundaries, i)
        w1 = corpus.text[lower:upper]
        w2 = corpus.text[lower:i]
        w3 = corpus.text[i:upper]

        if i in corpus.boundaries:
            corpus.boundaries.remove(i)

        # subtract 1 from the counts to compensate for counting itself
        utt = (len(corpus.utt_boundaries) if upper in corpus.utt_boundaries else
               n - len(corpus.utt_boundaries))
        p_h1 = ((words.count(w1) - 1 + alpha * corpus.p0(w1)) /
                (n + alpha)) * ((utt + 2) / (n + rho))

        p_h2 = ((words.count(w2) + alpha * corpus.p0(w2)) / (n + alpha) *
                ((n - len(corpus.utt_boundaries) + rho/2) / (n + rho)) *
                ((words.count(w3) + 1 if w2 == w3 else 0 + alpha *
                  corpus.p0(w3)) / (n + 1 + alpha)) *
                ((utt + 1 if w2 == w3 else 0 + rho/2) / (n + 1 + rho)))

        print '{}, {}'.format(p_h1, p_h2)
        if p_h2 > p_h1:
            corpus.boundaries.add(i)


def main():
    corpus = Corpus(argv[1])
    gibbs_iteration(corpus)
    print corpus.boundaries


if __name__ == "__main__":
    main()
