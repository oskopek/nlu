from typing import Optional

import numpy as np
import pandas as pd

from .preprocessing import Preprocessing
from .stories import StoriesDataset


class Datasets:

    def __init__(self,
                 train_file: str,
                 eval_file: str,
                 test_file: str,
                 preprocessing: Optional[Preprocessing] = None,
                 roemmele_multiplicative_factor: int = 0) -> None:
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.preprocessing = preprocessing
        self.roemmele_multiplicative_factor = roemmele_multiplicative_factor

        self._load()

    @staticmethod
    def _read(file: str) -> pd.DataFrame:
        df = pd.read_csv(file)
        return df

    @staticmethod
    # storyid, sentence1, sentence2, sentence3, sentence4, ending1, ending2, label
    def _read_train(file: str) -> pd.DataFrame:
        df = Datasets._read(file)
        del df['storytitle']
        df = df.rename(index=str, columns={"sentence5": "ending1"})
        df['ending2'] = pd.Series([""] * len(df), index=df.index)
        df['label'] = pd.Series([1] * len(df), index=df.index)
        return df

    @staticmethod
    def _read_eval(file: str) -> pd.DataFrame:
        df = Datasets._read(file)
        df = df.rename(
                index=str,
                columns={
                        "InputStoryid": "storyid",
                        "InputSentence1": "sentence1",
                        "InputSentence2": "sentence2",
                        "InputSentence3": "sentence3",
                        "InputSentence4": "sentence4",
                        "RandomFifthSentenceQuiz1": "ending1",
                        "RandomFifthSentenceQuiz2": "ending2",
                        "AnswerRightEnding": "label"
                })
        return df

    # TODO(oskopek): Sample random train endings per epoch.
    @staticmethod
    def _sample_random_train_endings(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assumes all `ending2`s are empty and all `label`s are 1.
        Also shuffles randomly (~Bernoulli(1/2)) endings so that about half of labels is 1 and half is 2.
        """

        def sample_without_current(length: int) -> np.ndarray:

            def has_identical(xs: np.ndarray) -> bool:
                res: int = np.sum(xs == np.arange(0, len(xs)))
                return res > 0

            array = np.random.randint(0, length, size=length)
            while has_identical(array):
                array = np.random.randint(0, length, size=length)
            return array

        sampled_indexes = sample_without_current(len(df))
        ending1 = df['ending1'].values
        ending2 = ending1[sampled_indexes]
        label = df['label'].values

        # Swap ending1 and ending2
        sampled_swap = np.random.choice([True, False], size=len(df))
        np.copyto(ending2, ending1, where=sampled_swap)
        ending2_orig = df.ix[sampled_indexes, 'ending1'].values
        np.copyto(ending1, ending2_orig, where=sampled_swap)
        np.place(label, sampled_swap, [2])

        df['ending1'] = ending1
        df['ending2'] = ending2
        df['label'] = label
        return df

    # TODO(oskopek): Sample random train endings per epoch.
    @staticmethod
    def _sample_random_train_endings_roemmele(df: pd.DataFrame, multiplicative_factor: int = 4) -> pd.DataFrame:
        """
        Assumes all `ending2`s are empty and all `label`s are 1.
        """

        def sample_without_current(length: int) -> np.ndarray:

            def has_identical(xs: np.ndarray) -> bool:
                res: int = np.sum(xs == np.arange(0, len(xs)))
                return res > 0

            array = np.random.randint(0, length, size=length)
            while has_identical(array):
                array = np.random.randint(0, length, size=length)
            return array

        dfs = []
        for i in range(multiplicative_factor):
            sampled_indexes = sample_without_current(len(df))
            df2 = df.copy(deep=True)
            df2['storyid'] = df2[['storyid']].applymap(lambda idx: f"{idx}_{i}")['storyid'].values
            df2['ending1'] = df.ix[sampled_indexes, 'ending1'].values
            df2['label'] = np.zeros_like(df2['label'].values)
            dfs.append(df2)
        df = df.append(dfs)
        return df

    def _load(self) -> None:
        print("Loading data from disk...", flush=True)
        df_train = self._read_train(self.train_file)
        df_eval = self._read_eval(self.eval_file)
        df_test = None
        if self.test_file:
            df_test = self._read_eval(self.test_file)

        print("Sampling random train endings...", flush=True)
        if self.roemmele_multiplicative_factor is not None:
            label_vocab = {0: 0, 1: 1}
            df_train = Datasets._sample_random_train_endings_roemmele(df_train, self.roemmele_multiplicative_factor)
        else:
            label_vocab = {1: 0, 2: 1}
            df_train = Datasets._sample_random_train_endings(df_train)

        print("Pre-processing...", flush=True)
        if self.preprocessing:
            self.preprocessing.transform(df_train, evaluate=False)
            self.preprocessing.transform(df_eval, evaluate=True)
            if self.test_file:
                self.preprocessing.transform(df_test, evaluate=True)

        print("Generating TF data...", flush=True)
        self.train = StoriesDataset(df_train, vocabularies=None, label_dictionary=label_vocab)
        self.eval = StoriesDataset(df_eval, vocabularies=self.train.vocabularies)
        if self.test_file:
            self.test = StoriesDataset(df_test, vocabularies=self.train.vocabularies)
