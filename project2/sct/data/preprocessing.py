import collections
import re
from typing import Any, Callable, List, Tuple, Dict, Optional, Union, Counter

import nltk
import pandas as pd

from .utils import MissingDict, PAD_TOKEN, UNK_TOKEN

ArgsType = Any
PreprocessingMethod = Callable[[pd.DataFrame, ArgsType, bool], pd.DataFrame]
Vocab = Dict[str, int]
InvVocab = Dict[int, str]


# TODO(oskopek): Add support for external vocabularies.
# TODO(oskopek): Add implementation for the evaluation=True flag.
class Preprocessing:
    BASE_VOCAB: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    def __init__(self,
                 standardize: bool = False,
                 contractions: bool = False,
                 rem_numbers: bool = False,
                 punct_squash: bool = False,
                 rem_stopwords: bool = False,
                 stemming: Optional[nltk.stem.StemmerI] = None,
                 lemmatization: Optional[nltk.stem.WordNetLemmatizer] = None,
                 cut_size: Optional[int] = None,
                 sentence_indexer: List[str] = list()) -> None:
        self.sentence_indexer = sentence_indexer
        self.methods: List[Tuple[PreprocessingMethod, ArgsType]] = [
                (self._standardize, standardize), (self._contractions, contractions), (self._rem_numbers, rem_numbers),
                (self._punct_squash, punct_squash), (self._rem_stopwords, rem_stopwords), (self._stemming, stemming),
                (self._lemmatization, lemmatization), (self._cut_sentences, cut_size)
        ]

    def transform(self, df: pd.DataFrame, evaluate: bool = False) -> pd.DataFrame:
        for fn, args in self.methods:
            if args:
                df = fn(df, args, evaluate)
        return df

    def _map_df(self, df: pd.DataFrame, fn: Callable[[str], str]) -> pd.DataFrame:
        return df[self.sentence_indexer].applymap(fn)  # ['sentence1', ..., 'ending2']

    def _map_words(self, df: pd.DataFrame, fn: Callable[[List[str]], List[str]]) -> pd.DataFrame:
        return self._map_df(df, lambda line: " ".join(fn(line.split())))

    def _contractions(self, df: pd.DataFrame, args: ArgsType, evaluate: bool = False):
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
        pattern_map = [(re.compile(x), y) for x, y in re_map]

        def contractions(line: str) -> str:
            for reg, subs in pattern_map:
                line = re.sub(reg, subs, line)
            return line

        return self._map_df(df, contractions)

    def _standardize(self, df: pd.DataFrame, args: ArgsType, evaluate: bool = False):
        df = self._map_df(df, lambda line: ' '.join(nltk.word_tokenize(line)))
        return self._map_words(df, lambda words: [word.strip() for word in words])

    def _rem_numbers(self, df: pd.DataFrame, args: ArgsType, evaluate: bool = False):
        re_map = [
                (r" [0-9]+ ", " "),
                (r"[0-9]+", " "),
        ]
        pattern_map = [(re.compile(x), y) for x, y in re_map]

        def rem_numbers(line: str) -> str:
            for reg, subs in pattern_map:
                line = re.sub(reg, subs, line)
            return line

        return self._map_df(df, rem_numbers)

    def _punct_squash(self, df: pd.DataFrame, args: ArgsType, evaluate: bool = False) -> pd.DataFrame:
        pattern = re.compile(r"([^a-z0-9] ?)\1+")
        repl = r" \1 "
        return self._map_df(df, lambda line: re.sub(pattern, repl, line))

    def _rem_stopwords(self, df: pd.DataFrame, args: ArgsType, evaluate: bool = False) -> pd.DataFrame:
        stop_words = set(nltk.corpus.stopwords.words('english'))
        return self._map_words(df, lambda words: [word for word in words if word not in stop_words])

    def _stemming(self, df: pd.DataFrame, stemmer: nltk.stem.StemmerI, evaluate: bool = False) -> pd.DataFrame:
        return self._map_words(df, lambda words: [stemmer.stem(word) for word in words])

    def _cut_sentences(self, df: pd.DataFrame, cut_size: int, evaluate: bool = False) -> pd.DataFrame:
        assert cut_size is not None and cut_size > 0
        return self._map_words(df, lambda words: words[:cut_size])

    def _lemmatization(self, df: pd.DataFrame, lemmatizer: nltk.stem.WordNetLemmatizer,
                       evaluate: bool = False) -> pd.DataFrame:
        return self._map_words(df, lambda words: [lemmatizer.lemmatize(word) for word in words])

    def _vocab_downsize_to(self, df: pd.DataFrame, vocab, inv_vocab, evaluate: bool = False) -> pd.DataFrame:
        return self._map_words(df, lambda words: [inv_vocab[vocab[word]] for word in words])

    def _vocab_downsize_vocab(self, df: pd.DataFrame, vocab_size: int, evaluate: bool = False) -> Dict[str, int]:
        counter: Counter[str] = collections.Counter()
        for key in self.sentence_indexer:
            for line in df[key]:
                counter.update(line.split())

        vocab = dict(self.BASE_VOCAB)
        for word, _ in counter.most_common(vocab_size - len(self.BASE_VOCAB)):
            assert word not in vocab
            vocab[word] = len(vocab)

        return MissingDict(vocab, default_val=vocab[UNK_TOKEN])

    def vocab(self, df: pd.DataFrame, vocab_downsize: Union[Tuple[Vocab, InvVocab], int],
              evaluate: bool = False) -> Union[pd.DataFrame, Tuple[Vocab, InvVocab]]:
        if isinstance(vocab_downsize, int):
            vocab = self._vocab_downsize_vocab(df, vocab_downsize)
            inv_vocab = {v: k for k, v in vocab.items()}
            return vocab, inv_vocab
        else:
            return self._vocab_downsize_to(df, *vocab_downsize)
