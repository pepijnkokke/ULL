import sys
import re
from segment import *


def precision_recall_f1(trained, reference):
    correct = 0
    for t, r in zip(trained, reference):
        if t == 1 and r == 1:
            correct += 1

    precision = float(correct) / len([x for x in trained if x == 1])
    recall = float(correct) / len([x for x in reference if x == 1])
    f = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f


def lexicon_precision_recall(trained_words, reference_words):
    correct = len([w for w in trained_words if w in reference_words])

    precision = float(correct) / len(trained_words)
    recall = float(correct) / len(reference_words)
    f = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f


raw_data = sys.argv[1]
reference_data = sys.argv[2]

with open(sys.argv[3], 'r') as f:
    test_boundaries = eval(f.read())

with open(reference_data, 'r') as f:
    reference_text = f.read()

corpus = Corpus(raw_data)

# truncate the test boundaries, which still include the training boundaries
size = len(corpus.boundaries)
if (len(test_boundaries) > size):
    test_boundaries = test_boundaries[len(test_boundaries) - size + 1:]

reference_boundaries = [0] * size
n_bounds = 0
for i, ch in enumerate(reference_text):
    if ch == ' ':
        reference_boundaries[i - n_bounds] = 1
        n_bounds += 1

    if ch == '\n':
        n_bounds += 1

p, r, f = precision_recall_f1(test_boundaries, reference_boundaries)
print 'Ambiguous boundaries:'
print 'Precision:\t{:.2f}\nRecall:\t\t{:.2f}\nF1:\t\t{:.2f}'.format(p, r, f)

trained_words = get_words(corpus.text, list_or(corpus.utt_boundaries, test_boundaries))
reference_words = reference_text.replace('\n', ' ').split(' ')

p, r, f = lexicon_precision_recall(trained_words, reference_words)
print 'Lexicon:'
print 'Precision:\t{:.2f}\nRecall:\t\t{:.2f}\nF1:\t\t{:.2f}'.format(p, r, f)
