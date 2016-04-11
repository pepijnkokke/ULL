#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <map>
#include <boost/dynamic_bitset.hpp>

using namespace std;
using namespace boost;

class Corpus {
public:
    string text;
    dynamic_bitset<> utt_boundaries;
    dynamic_bitset<> boundaries;
    map<char, double> p_phonemes;

    Corpus(const char *path)
    {
        load_file(path);
        compute_p_phonemes();
    }

    double p0(string word, double p_hash=0.5)
    {
        double p = 1;
        for (char ph : word)
            p *= p_phonemes[ph];

        return p_hash * pow(1 - p_hash, word.size() - 1) * p;
    }

    vector<string> get_words()
    {
        vector<string> words;
        dynamic_bitset<> bounds = utt_boundaries | boundaries;

        for (int i = 0; i < text.size(); ++i) {
            if (bounds[i])
                words.push_back(string());

            words.back().append(1, text[i]);
        }

        return words;
    }

private:
    void load_file(const char *path)
    {
        fstream in(path);
        string line;
        stringstream text_stream;
        vector<int> utt_indices = {0};

        while (getline(in, line)) {
            text_stream << line;
            utt_indices.push_back(utt_indices.back() + line.size());
        }

        in.close();

        text = text_stream.str();

        utt_boundaries = dynamic_bitset<>(text.size() + 1);
        boundaries = dynamic_bitset<>(text.size() + 1);

        for (int index : utt_indices) {
            utt_boundaries[index] = 1;
        }
    }

    void compute_p_phonemes()
    {
        for (char phoneme : text) {
            // only compute if the char isn't in the map yet
            if (p_phonemes.find(phoneme) == p_phonemes.end())
                p_phonemes[phoneme] = count(text.begin(), text.end(), phoneme) /
                    (double)text.size();
        }
    }
};

map<string, int> histogram(const vector<string> &vec)
{
    map<string, int> counts;
    vector<string> unique_words;

    unique_copy(vec.begin(), vec.end(), back_inserter(unique_words));

    for (const string &word : unique_words) {
        if (counts.find(word) == counts.end())
            counts[word] = 1;
        else
            counts[word] += 1;
    }

    return counts;
}

void find_enclosing_boundaries(dynamic_bitset<> bounds, int i,
                               int *lower, int *upper)
{
    *lower = i - 1;
    while (!bounds[*lower])
        *lower -= 1;

    *upper = i + 1;
    while (!bounds[*upper])
        *upper += 1;
}

void gibbs_iteration(Corpus &corpus, double rho=2.0, double alpha=0.5)
{
    dynamic_bitset<> bounds = corpus.utt_boundaries | corpus.boundaries;
    vector<string> words = corpus.get_words();
    map<string, int> word_counts = histogram(words);

    for (int i = 0; i < corpus.text.size(); ++i) {
        char phoneme = corpus.text[i];

        if (corpus.utt_boundaries[i])
            continue;

        int lower, upper;
        find_enclosing_boundaries(bounds, i, &lower, &upper);

        string w1 = corpus.text.substr(lower, (upper - lower));
        string w2 = corpus.text.substr(lower, (i - lower));
        string w3 = corpus.text.substr(i, (upper - i));

        double n_;
        if (!bounds[i])
            n_ = words.size();
        else
            n_ = words.size() - 1;
        
        double n_dollar = corpus.utt_boundaries.count() - 1;
        double nu = corpus.utt_boundaries[upper] ? n_dollar : n_ - n_dollar;

        double p_h1_factor1;
        if (!bounds[i])
            p_h1_factor1 = (word_counts[w1] - 1 + alpha * corpus.p0(w1)) / (n_ + alpha);
        else
            p_h1_factor1 = (word_counts[w1] + alpha * corpus.p0(w1)) / (n_ + alpha);

        double p_h1_factor2 = (nu + rho/2) / (n_ + rho);

        double p_h2_factor1, p_h2_factor3;
        if (!bounds[i]) {
            p_h2_factor1 = (word_counts[w2] + alpha * corpus.p0(w2)) / (n_ + alpha);
            p_h2_factor3 = ((word_counts[w3] + (w2 == w3 ? 1 : 0) + alpha *
                             corpus.p0(w3)) / (n_ + 1 + alpha));
        } else {
            p_h2_factor1 = (word_counts[w2] - 1 + alpha * corpus.p0(w2)) / (n_ + alpha);
            p_h2_factor3 = ((word_counts[w3] - 1 + (w2 == w3 ? 1 : 0) + alpha *
                             corpus.p0(w3)) / (n_ + 1 + alpha));
        }

        double p_h2_factor2 = (n_ - n_dollar + rho/2) / (n_ + rho);
        double p_h2_factor4 = ((nu + (w2 == w3 ? 1 : 0) + rho/2) / (n_ + 1 + rho));

        double p_h1 = p_h1_factor1 * p_h1_factor2;
        double p_h2 = p_h2_factor1 * p_h2_factor2 * p_h2_factor3 * p_h2_factor4;

        if (p_h2 > p_h1)
            corpus.boundaries[i] = 1;
        else 
            corpus.boundaries[i] = 0;
    }
}

void write_boundaries(const Corpus &corpus, const char *filename)
{
    ofstream out(filename);
    out << "[";

    for (int i = 0; i < corpus.text.size(); ++i) {
        if (i != 0)
            out << ", ";
        out << corpus.boundaries[i];
    }
    
    out << "]";
    out.close();
}

int main(int argc, char *argv[])
{
    Corpus corpus(argv[1]);

    int n = stoi(argv[2]);
    for (int i = 0; i < n; ++i) {
        cout << "Iteration " << i << endl;
        gibbs_iteration(corpus);
    }

    write_boundaries(corpus, "boundaries.txt");
    for (auto &word : corpus.get_words())
        cout << word << endl;

    return 0;
}
