from collections import Counter
import re
from typing import Any, Callable, List, Tuple, Dict, Optional

import numpy as np
import nltk
import pandas as pd

# from nltk.stem.api import StemmerI
# from nltk.stem.regexp import RegexpStemmer
# from nltk.stem.lancaster import LancasterStemmer
# from nltk.stem.isri import ISRIStemmer
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.rslp import RSLPStemmer

from .utils import MissingDict

LinesType = pd.DataFrame
ArgsType = Any


class Preprocessing:
    methods: List[Tuple[Callable[[LinesType, ArgsType], LinesType], ArgsType]] = None

    PAD_SYMBOL: str = "$pad$"
    UNK_SYMBOL: str = "$unk$"
    BASE_VOCAB: Dict[str, int] = {PAD_SYMBOL: 0, UNK_SYMBOL: 1}

    def __init__(
            self,
            standardize: bool = True,
            contractions: bool = True,
            rem_numbers: bool = True,
            punct_squash: bool = True,
            rem_stopwords: bool = False,
            stemming: Optional[nltk.stem.StemmerI] = nltk.stem.PorterStemmer(),
            #  lemmatization: Optional[nltk.stem.WordNetLemmatizer] = nltk.stem.WordNetLemmatizer(),
            lemmatization: Optional[nltk.stem.WordNetLemmatizer] = None,
            padding_size=40) -> None:
        self.padding_size = padding_size
        self.methods = [  # line operations
                (self.standardize, standardize),
                (self.contractions, contractions),
                (self.rem_numbers, rem_numbers),
                (self.punct_squash, punct_squash),
                (self.lines_to_matrix, True),

                # matrix operations
                (self.rem_stopwords, rem_stopwords),
                (self.stemming, stemming),
                (self.lemmatization, lemmatization)
        ]

    def transform(self, lines: LinesType, evaluate: bool = False):
        for fn, args in self.methods:
            if args:
                lines = fn(lines, args)
        return lines

    def contractions(self, lines: LinesType, args: ArgsType):
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

        return contraction_map(lines)

    def standardize(self, lines: LinesType, args: ArgsType):

        def _standardize(lines):
            for line in lines:
                newline = line.strip().split()
                newline = " ".join([w.strip().lower() for w in newline])
                yield newline

        return _standardize(lines)

    def rem_numbers(self, lines: LinesType, args: ArgsType):
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

        return num_map(lines)

    def lines_to_matrix(self, lines: LinesType, args: ArgsType):
        lines = list(lines)
        if labels:
            if len(lines) != len(labels):
                print("Lines", len(lines), "labels", len(labels))
                assert len(lines) != len(labels)
        for i, line in enumerate(lines):
            lines[i] = line.split()
        return lines

    def punct_squash(self, lines: LinesType, args: ArgsType) -> LinesType:
        pattern = re.compile(r"([^a-z0-9] ?)\1+")
        repl = r" \1 "

        def gen_punct_squash(lines):
            for line in lines:
                yield re.sub(pattern, repl, line)

        return gen_punct_squash(lines)

    def rem_stopwords(self, lines: LinesType, args: ArgsType) -> LinesType:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        for i, line in enumerate(lines):
            new_line = []
            for word in line:
                if word not in stop_words:
                    new_line.append(word)
            lines[i] = new_line
        return lines

    def stemming(self, lines: LinesType, stemmer) -> LinesType:
        for i, line in enumerate(lines):
            new_line = []
            for word in line:
                stemmed = stemmer.stem(word)
                new_line.append(stemmed)
            lines[i] = new_line
        return lines

    def lemmatization(self, lines: LinesType, lemmatizer) -> LinesType:
        for i, line in enumerate(lines):
            new_line = []
            for word in line:
                lemma = lemmatizer.lemmatize(word)
                new_line.append(lemma)
            lines[i] = new_line
        return lines

    def _vocab_downsize_dict(self, lines: LinesType, vocab, inv_vocab):
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

    def _vocab_downsize_tosize(self, lines: LinesType, vocab_size):
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

    def vocab(self, lines: LinesType, vocab_downsize):
        if isinstance(vocab_downsize, int):
            vocab = self._vocab_downsize_tosize(lines, vocab_downsize)
            inv_vocab = {v: k for k, v in vocab.items()}
            return vocab, inv_vocab
        else:
            return self._vocab_downsize_dict(lines, *vocab_downsize)
