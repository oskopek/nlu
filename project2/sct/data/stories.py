from typing import *

from dotmap import DotMap
import numpy as np
import pandas as pd

from .utils import MissingDict, UNK_TOKEN, BASE_VOCAB

T = TypeVar('T')
Char = TypeVar('Char', chr)
Word = TypeVar('Word', str)
Sentence = TypeAlias(Sequence[Word])
IntSentence = TypeAlias(Sequence[int])
DatasetRow = Tuple(Sentence, Sentence, Sentence, Sentence, Sentence, Sentence)
Dataset = Sequence[DatasetRow]


def generate_balanced_permutation(labels: Sequence[T], batch_size: int = 1, shuffle: bool = True) -> Sequence[int]:
    """Currently only works for binary labels `{1, 2}` and an approximately balanced dataset."""
    if not shuffle:
        return np.arange(len(labels))

    permutation = np.random.permutation(len(labels))  # shuffle all
    label_split = {1: [], 2: []}  # split by label
    for i, label in enumerate(labels):
        label_split[label].append(permutation[i])

    balanced_perm = np.zeros_like(permutation)
    ones_ratio = len(label_split[1]) / len(labels)  # balance batches by ratio
    for n_batch in range(0, len(labels), batch_size):
        counts = {1: 0, 2: 0}
        for n_sample in range(batch_size):
            cur_ones_ratio = counts[1] / (counts[1] + counts[2])
            if cur_ones_ratio < ones_ratio:
                chosen_label = 1
            elif cur_ones_ratio > ones_ratio:
                chosen_label = 2
            else:
                chosen_label = np.random.choice(label_split.keys(), 1)
            balanced_perm[n_batch * batch_size + n_sample] = label_split[chosen_label].pop()
            counts[chosen_label] += 1
    return balanced_perm


class Vocabularies:
    sentence_vocabulary: Dict[Sentence, int] = None
    word_vocabulary: Dict[Word, int] = None
    char_vocabulary: Dict[Char, int] = None

    @staticmethod
    def _default_dict() -> MissingDict[str, int]:
        return MissingDict(BASE_VOCAB, default_val=BASE_VOCAB[UNK_TOKEN])

    def __init__(self, dataset: Sequence[DatasetRow]):
        self.sentence_vocabulary = Vocabularies._default_dict()
        self.word_vocabulary = Vocabularies._default_dict()
        self.char_vocabulary = Vocabularies._default_dict()

        for row in dataset:
            for sentence in row:
                if sentence not in self.sentence_vocabulary:
                    self.sentence_vocabulary[sentence] = len(self.sentence_vocabulary)
                for word in sentence:
                    if word not in self.word_vocabulary:
                        self.word_vocabulary[word] = len(self.word_vocabulary)
                    for char in word:
                        if char not in self.char_vocabulary:
                            self.char_vocabulary[char] = len(self.char_vocabulary)


class NLPData:
    sentence_ids: List[List[int]] = None  # dataset_idx -> [sentence_id]
    word_ids: List[List[int]] = None  # sentence_id -> [word_id]
    char_ids: List[List[int]] = None  # word_id -> [char_id]

    @staticmethod
    def _create_ids(parent_vocab, cur_vocab) -> List[List[int]]:
        ids = [[]] * len(parent_vocab)
        for word, word_idx in parent_vocab.items():
            assert word_idx < len(parent_vocab)
            ids[word_idx] = [cur_vocab[char] for char in word]
        return ids

    def __init__(self, dataset: Sequence[DatasetRow], vocabularies: Vocabularies):
        self.sentence_ids = []
        for row in dataset:
            row_sentence_ids = []
            for sentence in row:
                sentence_id = vocabularies.sentence_vocabulary[sentence]
                row_sentence_ids.append(sentence_id)
            self.sentence_ids.append(row_sentence_ids)

        self.word_ids = NLPData._create_ids(vocabularies.sentence_vocabulary, vocabularies.word_vocabulary)
        self.char_ids = NLPData._create_ids(vocabularies.word_vocabulary, vocabularies.char_vocabulary)

    def __len__(self):
        return len(self.sentence_ids)

    def batch_iterator(self, permutation: Sequence[int], batch_size: int = 1,
                       padding: int = 0) -> Generator[Sequence[np.ndarray]]:
        assert len(self) == len(permutation)
        for i in range(0, len(self), batch_size):
            batch_idxs = permutation[i:i + batch_size]
            if len(batch_idxs) == 0:
                raise StopIteration
            batch_sentence_ids = [self.sentence_ids[idx] for idx in batch_idxs]
            batch_word_ids, batch_sentence_lens = self._create_batch(batch_sentence_ids, )

            max_word_len = max(len(seq) for seq in batch_word_ids)

            words = np.full((len(batch_idxs), max_seq_len), fill_value=padding, dtype=np.int32)
            word_lens = np.zeros(len(batch_idxs), dtype=np.int32)
            for j, idx in enumerate(batch_idxs):
                seq = self.sentences[idx]
                word_lens[j] = len(seq)
                words[j, :word_lens[j]] = np.asarray(seq)

            yield (words, word_lens)


class StoriesDataset:
    SENTENCES = 4
    ENDINGS = 2
    TOTAL_SENT = SENTENCES + ENDINGS

    story_ids: Sequence[str] = None
    sentences: NLPData = None
    labels: Sequence[int] = None
    _len: int = None

    def __init__(self, df: pd.DataFrame, vocabularies: Vocabularies = None) -> None:
        self._len = len(df)
        self.story_ids = df['storyid'].values
        self.labels = df['labels'].values

        dataset = self._create_nlp_text_dataset(df)
        del df

        if vocabularies is None:
            self.vocabularies = Vocabularies(dataset)
        else:
            self.vocabularies = vocabularies

        self.sentences = NLPData(dataset, vocabularies=self.vocabularies)

    def __len__(self) -> int:
        return self._len

    def _create_nlp_text_dataset(self, df):
        # Index into Pandas DataFrame
        sentence_indexer = [f"sentence{i+1}" for i in range(self.SENTENCES)] + [f"ending{i+1}" for i in range(
            self.ENDINGS)]
        # Split words in columns
        columns = [[sentence.strip().split() for sentence in df[key].values] for key in sentence_indexer]
        # Transpose:
        return [[column[i] for column in columns] for i in range(len(df))]

    @staticmethod
    def _sequence_batch_iterator(seq: Sequence[T], permutation: Sequence[int],
                                 batch_size: int = 1) -> Generator[Sequence[T]]:
        for i in range(0, len(seq), batch_size):
            batch = list()
            for idx in permutation[i:i + batch_size]:
                batch.append(seq[idx])
            yield batch

    def batch_per_epoch_generator(self, batch_size: int = 1,
                                  shuffle: bool = True) -> Generator[DotMap[str, Union[np.ndarray, bool]]]:
        permutation = generate_balanced_permutation(self.labels, batch_size=batch_size, shuffle=shuffle)
        sentences_it = self.sentences.batch_iterator(permutation, batch_size=batch_size)
        labels_it = StoriesDataset._sequence_batch_iterator(self.labels, permutation, batch_size=batch_size)
        story_ids_it = StoriesDataset._sequence_batch_iterator(self.story_ids, permutation, batch_size=batch_size)
        for n_batch in range(self.n_batches(batch_size=batch_size)):
            sentence_ids, word_ids, sentence_lens, char_ids, word_lens = next(sentences_it)
            assert sentence_ids.shape[0] == batch_size
            labels = next(labels_it)
            story_ids = next(story_ids_it)

            yield DotMap({
                    "sentence_ids": sentence_ids,
                    "word_ids": word_ids,
                    "sentence_lens": sentence_lens,
                    "char_ids": char_ids,
                    "word_lens": word_lens,
                    "labels": labels,
                    "story_ids": story_ids,
                    "is_training": shuffle,
            })

    def batches_per_epoch(self, batch_size: int = 1, shuffle: bool = True):
        n_batches = self.n_batches(batch_size=batch_size)
        batch_generator = self.batch_per_epoch_generator(batch_size=batch_size, shuffle=shuffle)
        return n_batches, batch_generator

    def n_batches(self, batch_size: int = 1):
        return len(self) // batch_size + len(self) % batch_size
