import re
from typing import *

import numpy as np
import nltk

# from nltk.stem.api import StemmerI
# from nltk.stem.regexp import RegexpStemmer
# from nltk.stem.lancaster import LancasterStemmer
# from nltk.stem.isri import ISRIStemmer
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.rslp import RSLPStemmer

from .utils import MissingDict


class Preprocessing:
    methods = None

    PAD_SYMBOL = "$pad$"
    UNK_SYMBOL = "$unk$"
    BASE_VOCAB = {PAD_SYMBOL: 0, UNK_SYMBOL: 1}

    def __init__(
            self,
            standardize=True,
            segment_hashtags=10,
            contractions=True,
            rem_numbers=True,
            punct_squash=True,
            fix_slang=True,
            word_squash=3,
            expl_negations=False,
            rem_stopwords=False,
            stemming=nltk.stem.PorterStemmer(),  #  stemming=None,
            #  lemmatization=nltk.stem.WordNetLemmatizer(),
            lemmatization=None,
            padding_size=40):
        self.padding_size = padding_size
        self.methods = [  # line operations
                (self.standardize, standardize),
                (self.word_squash, word_squash),
                (self.segment_hashtags, segment_hashtags),
                (self.fix_slang, fix_slang),
                (self.contractions, contractions),
                (self.rem_numbers, rem_numbers),
                (self.punct_squash, punct_squash),
                (self.lines_to_matrix, True),

                # matrix operations
                (self.expl_negations, expl_negations),
                (self.rem_stopwords, rem_stopwords),
                (self.stemming, stemming),
                (self.lemmatization, lemmatization)
        ]

    def transform(self, lines, eval: bool=False):  # labels == None => test transformation
        for fn, args in self.methods:
            # assert len(lines) == len(labels)
            if args:
                lines, labels = fn(lines, labels, args)
        return lines, labels

    def contractions(self, lines, labels, args):
        re_map = [
                (r" ([a-z]+)'re ", r" \1 are "),
                (r" youre ", " you are "),
                (r" (it|that|he|she|what|there|who|here|where|how)'s ", r" \1 is "),
                (r" i[ ]?'[ ]?m ", " i am "),
                (r" can't ", " can not "),
                (r" ain't ", " am not "),
                (r" won't ", " will not "),
                (r" ([a-z]+)n't ", r" \1 not "),
                (r" ([a-z]+)'ll ", r" \1 will "),
                (r" ([a-z]+)'ve ", r" \1 have "),
                (r" (i|you|he|she|it|we|they|u)'d ", r" \1 would "),
                (r" (how|why|where|what)'d ", r" \1 did "),
                (r" ([a-z]+)'d ", r" \1 "),  # just remove it here "lol'd"
        ]

        re_map = [(re.compile(x), y) for x, y in re_map]

        def contraction_map(lines):
            for line in lines:
                for reg, subs in re_map:
                    line = re.sub(reg, subs, line)
                yield line

        return contraction_map(lines), labels

    def standardize(self, lines, labels, args):

        def _standardize(lines):
            for line in lines:
                newline = line.strip().split()
                newline = " ".join([w.strip().lower() for w in newline])
                yield newline

        return _standardize(lines), labels

    def rem_numbers(self, lines, labels, args):
        re_map = [
                (r" [0-9]+ ", " "),
                (r"[0-9]+", " "),
        ]

        re_map = [(re.compile(x), y) for x, y in re_map]

        def num_map(lines):
            for line in lines:
                for reg, subs in re_map:
                    line = re.sub(reg, subs, line)
                yield line

        return num_map(lines), labels

    def lines_to_matrix(self, lines, labels, args):
        lines = list(lines)
        if labels:
            if len(lines) != len(labels):
                print("Lines", len(lines), "labels", len(labels))
                assert len(lines) != len(labels)
        for i, line in enumerate(lines):
            lines[i] = line.split()
        return lines, labels

    def punct_squash(self, lines, labels, args):
        pattern = re.compile(r"([^a-z0-9] ?)\1+")
        repl = r" \1 "

        def gen_punct_squash(lines):
            for line in lines:
                yield re.sub(pattern, repl, line)

        return gen_punct_squash(lines), labels

    def rem_stopwords(self, lines, labels, args):
        stop_words = set(nltk.corpus.stopwords.words('english'))

        def gen_stopwords(lines):
            for i, line in enumerate(lines):
                new_line = []
                for word in line:
                    if word not in stop_words:
                        new_line.append(word)
                lines[i] = new_line
            return lines

        return gen_stopwords(lines), labels

    def stemming(self, lines, labels, stemmer):

        def gen_stem(lines):
            for i, line in enumerate(lines):
                new_line = []
                for word in line:
                    stemmed = stemmer.stem(word)
                    new_line.append(stemmed)
                lines[i] = new_line
            return lines

        return gen_stem(lines), labels

    def lemmatization(self, lines, labels, lemmatizer):

        def gen_lemma(lines):
            for i, line in enumerate(lines):
                new_line = []
                for word in line:
                    lemma = lemmatizer.lemmatize(word)
                    new_line.append(lemma)
                lines[i] = new_line
            return lines

        return gen_lemma(lines), labels

    def _vocab_downsize_dict(self, lines, vocab, inv_vocab):
        lines = np.asarray(lines)
        data = np.full((len(lines), self.padding_size), "$pad$", dtype=object)
        cut_counter = 0
        for i, line in enumerate(lines):
            strs = np.asarray(line).astype(object)
            fin_len = min(self.padding_size, len(strs))
            data[i, :fin_len] = strs[:fin_len]
            if len(strs) > self.padding_size:
                cut_counter += 1
        if cut_counter > 0:
            print("WARNING: Cut {} sentences to length {}.".format(cut_counter, self.padding_size))

        data = np.vectorize(lambda word: inv_vocab[vocab[word]])(data)
        return data

    def _vocab_downsize_tosize(self, lines, vocab_size):
        counter = Counter()
        for line in lines:
            counter.update(line)

        vocab = dict(self.BASE_VOCAB)
        uid = len(self.BASE_VOCAB)

        for word, _ in counter.most_common(vocab_size - len(self.BASE_VOCAB)):
            assert word not in vocab
            vocab[word] = uid
            uid += 1

        return MissingDict(vocab, default_val=vocab[self.UNK_SYMBOL])

    def vocab(self, lines, vocab_downsize):
        if isinstance(vocab_downsize, int):
            vocab = self._vocab_downsize_tosize(lines, vocab_downsize)
            inv_vocab = {v: k for k, v in vocab.items()}
            return vocab, inv_vocab
        else:
            return self._vocab_downsize_dict(lines, *vocab_downsize)
