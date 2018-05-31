from typing import Optional

import numpy as np
import pandas as pd

from .preprocessing import Preprocessing
from .stories import NLPStoriesDataset, SkipThoughtStoriesDataset, StoriesDataset

from .skip_thoughts import configuration
from .skip_thoughts import encoder_manager


class Datasets:

    def __init__(self,
                 train_file: str,
                 eval_file: str,
                 test_file: str,
                 preprocessing: Optional[Preprocessing] = None,
                 roemmele_multiplicative_factor: int = 0,
                 eval_train: bool = False,
                 balanced_batches: bool = False,
                 sent_embedding: bool = False) -> None:
        self.train_file = train_file
        self.eval_file = eval_file
        self.test_file = test_file
        self.preprocessing = preprocessing
        self.roemmele_multiplicative_factor = roemmele_multiplicative_factor
        self.eval_train = eval_train
        self.balanced_batches = balanced_batches
        self.sent_embedding = sent_embedding

        PREFIX = '/cluster/scratch/oskopek'
        VOCAB_FILE = "{}/st/{}/vocab.txt"
        EMBEDDING_MATRIX_FILE = "{}/st/{}/embeddings.npy"
        CHECKPOINT_PATH = "{}/st/{}/model.ckpt-{}"

        self.encoder = encoder_manager.EncoderManager()
        self.encoder.load_model(
                configuration.model_config(),
                vocabulary_file=VOCAB_FILE.format(PREFIX, "uni"),
                embedding_matrix_file=EMBEDDING_MATRIX_FILE.format(PREFIX, "uni"),
                checkpoint_path=CHECKPOINT_PATH.format(PREFIX, "uni", 501424))
        self.encoder.load_model(
                configuration.model_config(bidirectional_encoder=True),
                vocabulary_file=VOCAB_FILE.format(PREFIX, "bi"),
                embedding_matrix_file=EMBEDDING_MATRIX_FILE.format(PREFIX, "bi"),
                checkpoint_path=CHECKPOINT_PATH.format(PREFIX, "bi", 500008))

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
    def _sample_random_train_ending2(df: pd.DataFrame) -> pd.DataFrame:
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
    def _sample_random_train_ending1_roemmele(df: pd.DataFrame, multiplicative_factor: int = 4) -> pd.DataFrame:
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

    @staticmethod
    def _make_eval_ending1(df: pd.DataFrame) -> pd.DataFrame:
        twos_idxs = df['label'].values == 2

        ending1 = df['ending1'].values
        ending1_orig = df['ending1'].values.copy()
        ending2 = df['ending2'].values
        label = df['label'].values

        # Swap ending1 and ending2
        np.copyto(ending1, ending2, where=twos_idxs)
        np.copyto(ending2, ending1_orig, where=twos_idxs)
        np.place(label, twos_idxs, [1])

        df['ending1'] = ending1
        df['ending2'] = ending2
        df['label'] = label

        df2 = df.copy(deep=True)
        df2['label'] = np.zeros_like(label)
        df2['ending1'] = df2['ending2']

        df['ending2'] = np.full_like(ending2, '')
        df2['ending2'] = np.full_like(ending2, '')

        df = df.append(df2)
        return df

    def _load(self) -> None:
        print("Loading data from disk...", flush=True)
        if self.eval_train:
            df_train = self._read_eval(self.eval_file)
            threshold = int(0.8 * len(df_train))
            df_train = df_train[:threshold]
            df_eval = self._read_eval(self.eval_file)[threshold:]

            print("Sampling eval endings...", flush=True)
            if self.roemmele_multiplicative_factor is not None:
                label_vocab = {0: 0, 1: 1}
                df_train = Datasets._make_eval_ending1(df_train)
            else:
                label_vocab = {1: 0, 2: 1}
        else:
            df_train = self._read_train(self.train_file)
            df_eval = self._read_eval(self.eval_file)

            print("Sampling random train endings...", flush=True)
            if self.roemmele_multiplicative_factor is not None:
                label_vocab = {0: 0, 1: 1}
                df_train = Datasets._sample_random_train_ending1_roemmele(df_train, self.roemmele_multiplicative_factor)
            else:
                label_vocab = {1: 0, 2: 1}
                df_train = Datasets._sample_random_train_ending2(df_train)

        df_test = None
        if self.test_file:
            df_test = self._read_eval(self.test_file)

        print("Pre-processing...", flush=True)
        if self.preprocessing:
            self.preprocessing.transform(df_train, evaluate=False)
            self.preprocessing.transform(df_eval, evaluate=True)
            if self.test_file:
                self.preprocessing.transform(df_test, evaluate=True)

        if self.sent_embedding:
            print("Generating TF sentence embedded data...", flush=True)
            self.train: StoriesDataset = SkipThoughtStoriesDataset(
                    df_train,
                    encoder=self.encoder,
                    label_dictionary=label_vocab,
                    balanced_batches=self.balanced_batches)
            self.eval: StoriesDataset = SkipThoughtStoriesDataset(
                    df_eval, encoder=self.encoder, balanced_batches=self.balanced_batches)
            if self.test_file:
                self.test: StoriesDataset = SkipThoughtStoriesDataset(
                        df_test, encoder=self.encoder, balanced_batches=self.balanced_batches)
        else:
            print("Generating TF word data...", flush=True)
            self.train: StoriesDataset = NLPStoriesDataset(
                    df_train, vocabularies=None, label_dictionary=label_vocab, balanced_batches=self.balanced_batches)
            self.eval: StoriesDataset = NLPStoriesDataset(
                    df_eval, vocabularies=self.train.vocabularies, balanced_batches=self.balanced_batches)
            if self.test_file:
                self.test: StoriesDataset = NLPStoriesDataset(
                        df_test, vocabularies=self.train.vocabularies, balanced_batches=self.balanced_batches)
