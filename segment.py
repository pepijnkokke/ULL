#!/usr/bin/env python2

from sys          import argv
from collections  import Counter
from operator     import mul
from random       import random
from copy         import copy
from cPickle      import dump, load
from os.path      import exists
from json         import load
from progress.bar import Bar
from argparse     import ArgumentParser


FILENAME = 'boundaries.pickle'
FILENAME_EVAL = 'eval.pickle'


class Corpus:
    """
    Represents a training corpus. Keeps track of the text, utterance boundaries,
    word boundaries, phoneme probabilities and word probabilities.

    Boundaries are represented as bitstrings (which are represented as lists of
    the same length as the text with the elements being 0 or 1).
    """
    def __init__(self, path):
        utt_boundaries = [0]
        self.text = ''

        with open(path, 'r') as f:
            for utterance in f:
                utterance = utterance.strip()
                self.text += utterance
                utt_boundaries.append(utt_boundaries[-1] +
                                      len(utterance))

        utt_bitstring = [0] * (len(self.text) + 1)
        boundary_bitstring = [0] * (len(self.text) + 1)
        for bound in utt_boundaries:
            utt_bitstring[bound] = 1

        self.utt_boundaries = utt_bitstring
        self.boundaries = boundary_bitstring

        # precompute the phoneme probabilities
        self.pPhonemes = {}
        for phoneme in set(self.text):
            self.pPhonemes[phoneme] = float(self.text.count(phoneme)) / len(self.text)

    def pPhoneme(self, phon):
        """ Compute the prior probability of a phoneme. """
        return self.pPhonemes[phon]

    def p0(self, word, p_hashtag=0.5):
        p = reduce(mul, [self.pPhonemes[ph] for ph in word])
        return p_hashtag * (1 - p_hashtag)**(len(word) - 1) * p


def get_words(text, boundaries):
    """
    Returns a list of words resulting from the given string being segmented by
    the given boundaries.
    """
    out = []
    for phoneme, i in zip(text, boundaries):
        if i == 1:
            out.append([])

        out[-1].append(phoneme)

    return [''.join(word) for word in out]


def list_or(xs, ys):
    """
    Returns a new bitstring which is the logical or of the two given bitstrings.
    """
    return [1 if x == 1 or y == 1 else 0 for x, y in zip(xs, ys)]


def evaluate(corpus_found, corpus_true):
    """ Evaluate precision, recall and F0 for:
        - Words (word boundaries placed correctly before and after the word)
        - Lexical (lexical types found)
        - Possibly Ambiguous Boundaries (rather than looking at words, look at just the boundary positions excluding the utterance boundaries, as they are already correct)
        Note that these are exactly the evaluations as done by Goldwater & friends
        corpus_found : the corpus we arrived at after Gibbs sampling
        corpus_true  : the correctly segmented corpus we received
    """
    #Note: currently only the boundaries return non-zero values. Words and lexical types still wonkling

    boundaries_found = list_or(corpus_found.utt_boundaries, corpus_found.boundaries)
    boundaries_true = list_or(corpus_true.utt_boundaries, corpus_true.boundaries)

    #Words
    words_found = get_words(corpus_found.text, boundaries_found)
    words_true = get_words(corpus_true.text, boundaries_true)
    P,R = precision_recall(words_found,words_true)
    F = f_zero(P,R)

    #Lexical types #Note: it's possible these should just be the second column in dict.txt
    lexical_found = set(get_words(corpus_found.text,boundaries_found))
    lexical_true = set(get_words(corpus_true.text,boundaries_true))
    LP,LR = precision_recall(lexical_found,lexical_true)
    LF = f_zero(LP,LR)

    #Possibly Ambiguous Boundaries
    BP,BR = precision_recall(corpus_found.boundaries, corpus_true.boundaries) #Note: utterance boundaries are ignored as they are unambiguous
    BF = f_zero(BP,BR)

    #How do you like my string boyz
    return "\nBP: " + str(BP) + "\nBR: " + str(BR) + "\nBF: " + str(BF)

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
    denom = precision+recall
    if denom == 0: #Note: yes, if precision+recall can be zero something's not working right
        return 0
    else:
        return (2*precision*recall)/denom


def find_enclosing_boundaries(boundaries, i):
    """ Find the nearest boundaries on both sides of i. """
    lower = i - 1
    while boundaries[lower] != 1:
        lower -= 1

    upper = i + 1
    while boundaries[upper] != 1:
        upper += 1

    return lower, upper


def gibbs_iteration(corpus, rho=2.0, alpha=0.5, p_hash=0.5):
    """
    Perform a single Gibbs sampling iteration over the corpus.
    Boundaries are updated in the given corpus.
    """
    # precompute the word counts outside of the phoneme loop for efficiency
    boundaries = list_or(corpus.utt_boundaries, corpus.boundaries)
    words = get_words(corpus.text, boundaries)
    word_counts = Counter(words)

    # create a progress bar
    bar = Bar('Evaluating boundaries', max=len(corpus.text))
    p0 = lambda w: corpus.p0(w, p_hash)

    for i, phoneme in enumerate(corpus.text):
        # utterance boundaries are unambiguous, so they can be skipped
        if corpus.utt_boundaries[i] == 1:
            bar.next()
            continue

        # calculate the words around the potential boundary (w1 when there's no
        # boundary, w2+w3 when there is a boundary)
        lower, upper = find_enclosing_boundaries(boundaries, i)
        w1 = corpus.text[lower:upper]
        w2 = corpus.text[lower:i]
        w3 = corpus.text[i:upper]

        h_utt_boundaries = corpus.utt_boundaries[:lower] + corpus.utt_boundaries[upper:]

        # since the word counts were precomputed, we have to substract 1 from
        # the couunts of either w1 or w2 and w3 based on wether or not there was
        # a boundary there at the time of computation
        if boundaries[i] == 0:
            n_ = len(words)
        else:
            n_ = len(words) - 1

        n_dollar = h_utt_boundaries.count(1) - 1
        nu = n_dollar if corpus.utt_boundaries[upper] == 1 else n_ - n_dollar

        if boundaries[i] == 0:
            p_h1_factor1 = (word_counts[w1] - 1 + alpha * p0(w1)) / (n_ + alpha)
        else:
            p_h1_factor1 = (word_counts[w1] + alpha * p0(w1)) / (n_ + alpha)

        p_h1_factor2 = (nu + rho/2) / (n_ + rho)

        if boundaries[i] == 0:
            p_h2_factor1 = (word_counts[w2] + alpha * p0(w2)) / (n_ + alpha)
            p_h2_factor3 = ((word_counts[w3] + 1 if w2 == w3 else 0 + alpha *
                             p0(w3)) / (n_ + 1 + alpha))
        else:
            p_h2_factor1 = (word_counts[w2] - 1 + alpha * p0(w2)) / (n_ + alpha)
            p_h2_factor3 = ((word_counts[w3] - 1 + 1 if w2 == w3 else 0 + alpha *
                             p0(w3)) / (n_ + 1 + alpha))

        p_h2_factor2 = (n_ - n_dollar + rho/2) / (n_ + rho)
        p_h2_factor4 = ((nu + 1 if w2 == w3 else 0 + rho/2) / (n_ + 1 + rho))

        p_h1 = p_h1_factor1 * p_h1_factor2
        p_h2 = p_h2_factor1 * p_h2_factor2 * p_h2_factor3 * p_h2_factor4

        if p_h2 > p_h1:
            corpus.boundaries[i] = 1
        else:
            corpus.boundaries[i] = 0

        bar.next()

    bar.finish()


def main():
    parser = ArgumentParser(description='Segment a given corpus')
    parser.add_argument('corpus', type=str, help="path to the corpus")
    parser.add_argument('true_corpus', type=str, help="path to the true corpus")
    parser.add_argument('--alpha', default=0.5, type=float, help="the alpha hyperparameter")
    parser.add_argument('--phash', default=0.5, type=float, help="the p# hyperparameter")
    parser.add_argument('-i', default=10, type=int, help="number of training epochs")

    args = parser.parse_args()

    corpus = Corpus(args.corpus)

    # run the specified number of iterations of Gibbs sampling, saving the
    # output after each iteration
    for i in range(args.i):
        print i
        gibbs_iteration(corpus, alpha=args.alpha, p_hash=args.phash)

        with open(FILENAME, 'wb') as f:
            print 'Saving data to {}'.format(FILENAME)
            dump(corpus.boundaries, f)

    # Evaluation (if true corpus is provided)
    corpus_true = Corpus(args.true_corpus)
    eval = evaluate(corpus, corpus_true)
    print eval


if __name__ == "__main__":
    main()
