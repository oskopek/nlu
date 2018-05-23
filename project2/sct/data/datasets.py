from typing import Optional, List

import numpy as np
import pandas as pd

from .preprocessing import Preprocessing
from .stories import StoriesDataset


class Datasets:

    def __init__(self,
                 train_file: str,
                 eval_file: str,
                 test_file: str,
                 preprocessing: Optional[Preprocessing] = Preprocessing()) -> None:
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.preprocessing = preprocessing

        self._load()

    @staticmethod
    def _read(file: str) -> pd.DataFrame:
        df = pd.read_csv(file)
        df = df[:200]  # TODO: REMOVE ME
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

    @staticmethod
    def _sample_random_train_endings(df: pd.DataFrame) -> None:
        """
        Assumes all `ending2`s are empty and all `label`s are 1.
        Also shuffles randomly (~Bernoulli(1/2)) endings so that about half of labels is 1 and half is 2.
        """

        def generate_mask(idx: int, indexes_len: int) -> np.ndarray:
            mask = np.ones(indexes_len)
            mask[idx] = 0
            mask /= indexes_len - 1
            return mask

        def sample_without_current(length: int) -> List[int]:
            res = []
            indexes = np.arange(0, length)
            for idx in indexes:
                mask = generate_mask(idx, len(indexes))
                sampled = np.random.choice(indexes, replace=True, p=mask)
                assert sampled != idx
                res.append(sampled)
            return res

        sampled_indexes = sample_without_current(len(df))
        sampled_swap = np.random.choice([True, False], size=len(df))
        for i in range(len(df)):
            df.ix[i, ['ending2']] = df['ending1'][sampled_indexes[i]]
            if sampled_swap[i]:  # swap ending1 and ending2
                df.ix[i, ['ending1']], df.ix[i, ['ending2']] = df['ending2'][i], df['ending1'][i]
                df.ix[i, ['label']] = 2

    def _load(self) -> None:
        print("Loading data from disk...")
        df_train = self._read_train(self.train_file)
        df_eval = self._read_eval(self.eval_file)
        df_test = None
        if self.test_file:
            df_test = self._read_eval(self.test_file)

        print("Sampling random train endings...")
        Datasets._sample_random_train_endings(df_train)

        print("Pre-processing...")
        if self.preprocessing:
            self.preprocessing.transform(df_train, evaluate=False)
            self.preprocessing.transform(df_eval, evaluate=True)
            if self.test_file:
                self.preprocessing.transform(df_test, evaluate=True)

        print("Generating TF data...")
        self.train = StoriesDataset(df_train, vocabularies=None)
        self.eval = StoriesDataset(df_eval, vocabularies=self.train.vocabularies)
        if self.test_file:
            self.test = StoriesDataset(df_test, vocabularies=self.train.vocabularies)
