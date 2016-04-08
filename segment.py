from sys         import argv
from collections import namedtuple
from operator    import mul
from random      import random
from copy        import copy
from IPython     import embed
from bitstring   import BitArray


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

        # store positions of utterance boundaries
        uttr_bound_pos = [0]

        self.text = ''

        # add each utterance (stripped) to the text
        with open(path, 'r') as f:
            for uttr in f:
                uttr        = uttr.strip()
                self.text += uttr
                uttr_bound_pos.append(uttr_bound_pos[-1] + len(uttr))

        uttr_bounds = BitArray(length=len(self.corpus + 1))
        uttr_bounds.setall(0)
        word_bounds = BitArray(length=len(self.corpus + 1))
        word_bounds.setall(0)

        for pos in uttr_bound_pos:
            uttr_bounds.set(1,pos)

        self.uttr_bounds = uttr_bounds
        self.word_bounds = word_bounds

    def pPhoneme(self, phon):
        """ Compute the prior probability of a phoneme. """
        return float(self.text.count(phon)) / len(self.text)

    def p0(self, word, p_hashtag=0.5):
        p = reduce(mul, map(self.pPhoneme, word))
        return p_hashtag * (1 - p_hashtag)**(len(word) - 1) * p


def get_words(text, bounds):
    out = []
    for phoneme, i in zip(text, bounds):
        if i == 1:
            out.append([])

        out[-1].append(phoneme)

    return [''.join(word) for word in out]


def list_or(xs, ys):
    return [1 if x == 1 or y == 1 else 0 for x, y in zip(xs, ys)]


def evaluate(segmented_found,segmented_true,lexicon_found,lexicon_true):
    """ Evaluate precision, recall and F0 for:
        - Words (word bounds placed correctly before and after the word)
        - Lexical (lexical types found)
        - Bounds (rather than looking at words, look at just the boundary positions excluding the utterance bounds, as they are already correct)
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


def find_enclosing_bounds(bounds, i):
    """ Find the nearest bounds on both sides of i """
    lower = i - 1
    while bounds[lower] != 1:
        lower -= 1

    upper = i + 1
    while bounds[upper] != 1:
        upper += 1

    return lower, upper


def gibbs_iteration(corpus, rho=2.0, alpha=0.5):
    for i, phoneme in enumerate(corpus.text):
        # utterance bounds are unambiguous
        if corpus.uttr_bounds[i] == 1:
            continue

        # we COULD _||_ them all together here, but simply checking both arrays
        # is probably faster
        bounds = list_or(corpus.uttr_bounds, corpus.word_bounds)
        lower, upper = find_enclosing_bounds(bounds, i)
        w1 = corpus.text[lower:upper]
        w2 = corpus.text[lower:i]
        w3 = corpus.text[i:upper]

        h_ = corpus.text[:lower] + corpus.text[upper:]
        h_bounds = bounds[:lower] + bounds[upper:]
        h_uttr_bounds = corpus.uttr_bounds[:lower] + corpus.uttr_bounds[upper:]

        h_words = get_words(h_, h_bounds)
        n_ = len(h_words) # TODO: number of words or number of unique words?

        bounds[i] = 0

        n_dollar = h_uttr_bounds.count(1) - 1
        nu = n_dollar if corpus.uttr_bounds[upper] == 1 else n_ - n_dollar

        p_h1_factor1 = (h_words.count(w1) + alpha * corpus.p0(w1)) / (n_ + alpha)

        p_h1_factor2 = (nu + rho/2) / (n_ + rho)

        p_h2_factor1 = (h_words.count(w2) + alpha * corpus.p0(w2)) / (n_ + alpha)
        p_h2_factor2 = (n_ - n_dollar + rho/2) / (n_ + rho)
        p_h2_factor3 = ((h_words.count(w3) + 1 if w2 == w3 else 0 + alpha *
                  corpus.p0(w3)) / (n_ + 1 + alpha))
        p_h2_factor4 = ((nu + 1 if w2 == w3 else 0 + rho/2) / (n_ + 1 + rho))

        p_h1 = p_h1_factor1 * p_h1_factor2
        p_h2 = p_h2_factor1 * p_h2_factor2 * p_h2_factor3 * p_h2_factor4

        print '{}: {:.2e}, {:.2e} {}'.format(i, p_h1, p_h2,
                                             '(adding boundary)' if p_h2 > p_h1 else '')

        if p_h2 > p_h1:
            corpus.word_bounds[i] = 1


def main():
    corpus = Corpus(argv[1])
    gibbs_iteration(corpus)
    print corpus.word_bounds


if __name__ == "__main__":
    main()
