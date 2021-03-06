from collections import OrderedDict
from typing import Sequence, TypeVar, Tuple, Dict, List, Iterator, Union, Optional

import os

import numpy as np
import pandas as pd

from .utils import MissingDict, UNK_TOKEN, BASE_VOCAB, invert_dict, create_sentence_indexer

from .skip_thoughts.encoder_manager import EncoderManager

T = TypeVar('T')
Char = str
Word = str
Sentence = Sequence[Word]
IntSentence = Sequence[int]
DatasetRow = Tuple[Sentence, ...]
Dataset = Sequence[DatasetRow]


def generate_permutation(length: int, shuffle: bool = True) -> Sequence[int]:
    if not shuffle:
        return np.arange(length)
    else:
        return np.random.permutation(length)


# TODO(oskopek): This might produce last batches unbalanced, which means that the eval error will be skewed?
def generate_balanced_permutation(labels: Sequence[T], batch_size: int = 1, shuffle: bool = True) -> Sequence[int]:
    """Currently only works for binary labels and an approximately balanced dataset."""
    if not shuffle:
        return np.arange(len(labels))

    permutation = np.random.permutation(len(labels))  # shuffle all
    label_split: Dict[T, List[int]] = dict()  # split by label
    for i, label in enumerate(labels):
        if label not in label_split:
            label_split[label] = []
        label_split[label].append(permutation[i])

    balanced_perm = np.zeros_like(permutation)
    label_list = list(label_split.keys())
    one_label, two_label = label_list
    ones_ratio = len(label_split[one_label]) / len(labels)  # balance batches by ratio
    for n_batch in range(0, len(labels), batch_size):
        left_elements = min(batch_size, len(labels) - n_batch)
        counts: Dict[T, int] = {k: 0 for k in label_list}
        for n_sample in range(left_elements):
            if len(label_split[one_label]) == 0:
                chosen_label = two_label
            elif len(label_split[two_label]) == 0:
                chosen_label = one_label
            else:
                denominator = sum(counts.values())
                cur_ones_ratio = counts[one_label] / denominator if denominator != 0 else 0
                if cur_ones_ratio < ones_ratio:
                    chosen_label = one_label
                elif cur_ones_ratio > ones_ratio:
                    chosen_label = two_label
                else:
                    chosen_label = np.random.choice(label_list)
            balanced_perm[n_batch + n_sample] = label_split[chosen_label].pop()
            counts[chosen_label] += 1
        # print("ones_ratio", counts[one_label] / sum(counts.values()))
    return balanced_perm


class Vocabularies:
    WORD_DIM = 620

    @staticmethod
    def _default_dict() -> MissingDict[str, int]:
        return MissingDict(BASE_VOCAB, default_val=BASE_VOCAB[UNK_TOKEN])

    def __init__(self, dataset: Sequence[DatasetRow], skip_thought_folder: str) -> None:
        self.sentence_vocabulary: Dict[Tuple[Word, ...], int] = MissingDict({(): 0, (UNK_TOKEN,): 1}, default_val=1)
        self.word_vocabulary: Dict[Word, int] = Vocabularies._default_dict()
        self.char_vocabulary: Dict[Char, int] = Vocabularies._default_dict()

        for row in dataset:
            for sentence in row:
                tuple_sentence = tuple(sentence)
                if tuple_sentence not in self.sentence_vocabulary:
                    self.sentence_vocabulary[tuple_sentence] = len(self.sentence_vocabulary)
                for word in sentence:
                    if word not in self.word_vocabulary:
                        self.word_vocabulary[word] = len(self.word_vocabulary)
                    for char in word:
                        if char not in self.char_vocabulary:
                            self.char_vocabulary[char] = len(self.char_vocabulary)

        print("Loading st word embeddings")
        word_embeddings = np.load(os.path.join(skip_thought_folder, 'uni', 'embeddings.npy'))
        vocab: Dict[str, int] = {}
        for i, line in enumerate(open(os.path.join(skip_thought_folder, 'uni', 'vocab.txt'), 'r')):
            word = line.strip()
            self.word_vocabulary[word] = len(self.word_vocabulary)
            vocab[word] = i
        self.we_matrix = np.zeros(shape=(len(self.word_vocabulary), self.WORD_DIM), dtype=np.float32)

        words_missed = 0
        for word, word_index in self.word_vocabulary.items():
            if word in vocab:
                self.we_matrix[word_index] = word_embeddings[vocab[word]]
            else:
                print(word)
                words_missed += 1
        del word_embeddings, vocab
        print(f"Missed {words_missed}/{len(self.word_vocabulary)} words from word2vec embeddings.")


class NLPData:

    @staticmethod
    def _create_ids(parent_vocab, cur_vocab) -> List[List[int]]:
        ids: List[List[int]] = [[]] * len(parent_vocab)
        for word, word_idx in parent_vocab.items():
            assert word_idx < len(parent_vocab)
            ids[word_idx] = [cur_vocab[char] for char in word]
        return ids

    def __init__(self, dataset: Sequence[DatasetRow], vocabularies: Vocabularies) -> None:
        assert len(dataset) > 0
        row_len = len(dataset[0])

        # dataset_idx -> [sentence_id], matrix
        self.sentence_ids: np.ndarray = np.zeros((len(dataset), row_len), dtype=np.int32)
        for i, row in enumerate(dataset):
            assert len(row) == row_len
            row_sentence_ids = [vocabularies.sentence_vocabulary[tuple(sentence)] for sentence in row]
            self.sentence_ids[i] = np.asarray(row_sentence_ids, dtype=np.int32)

        # sentence_id -> [word_id]
        self.word_ids: List[List[int]] = NLPData._create_ids(vocabularies.sentence_vocabulary,
                                                             vocabularies.word_vocabulary)
        # word_id -> [char_id]
        self.char_ids: List[List[int]] = NLPData._create_ids(vocabularies.word_vocabulary, vocabularies.char_vocabulary)

    def __len__(self) -> int:
        return len(self.sentence_ids)

    @staticmethod
    def _create_sentence_tensors(
            batch_to_sentence_ids: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[int, int]]:
        flat_sentence_ids = list(set(idx for sentence in batch_to_sentence_ids for idx in sentence))
        sentence_dict = {k: v for v, k in enumerate(flat_sentence_ids)}

        batch_to_sentences = np.zeros_like(batch_to_sentence_ids)
        for i in range(batch_to_sentence_ids.shape[0]):
            for j in range(batch_to_sentence_ids.shape[1]):
                batch_to_sentences[i, j] = sentence_dict[batch_to_sentence_ids[i, j]]

        return (batch_to_sentence_ids, batch_to_sentences), sentence_dict

    @staticmethod
    def _create_token_tensors(word_ids, sentence_dict: Dict[int, int],
                              padding: int = 0) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[int, int]]:
        inv_sentence_dict = invert_dict(sentence_dict)
        flat_sentence_ids = [inv_sentence_dict[i] for i in range(len(sentence_dict))]
        sentence_lens = np.fromiter((len(word_ids[idx]) for idx in flat_sentence_ids), dtype=np.int32)
        max_sentence_len = np.max(sentence_lens)
        sentence_to_word_ids = np.full((len(flat_sentence_ids), max_sentence_len), fill_value=padding, dtype=np.int32)
        sentence_to_words = np.full((len(flat_sentence_ids), max_sentence_len), fill_value=padding, dtype=np.int32)
        unique_word_ids = list(set(word_id for sent_idx in flat_sentence_ids for word_id in word_ids[sent_idx]))
        word_dict = {k: v for v, k in enumerate(unique_word_ids)}
        for i, idx in enumerate(flat_sentence_ids):
            sentence, length = word_ids[idx], sentence_lens[i]
            sentence_to_word_ids[i, :length] = np.asarray(sentence)
            sentence_to_words[i, :length] = np.fromiter((word_dict[word_id] for word_id in sentence), dtype=np.int32)
        return (sentence_to_word_ids, sentence_to_words, sentence_lens), word_dict

    def _create_word_tensors(self, sentence_dict: Dict[int, int],
                             padding: int = 0) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[int, int]]:
        word_tensors, word_dict = NLPData._create_token_tensors(self.word_ids, sentence_dict, padding=padding)
        return word_tensors, word_dict

    def _create_char_tensors(self, word_dict: Dict[int, int],
                             padding: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        char_tensors, char_dict = NLPData._create_token_tensors(self.char_ids, word_dict, padding=padding)
        return char_tensors

    def batch_iterator(self, permutation: Sequence[int], batch_size: int = 1,
                       padding: int = 0) -> Iterator[Tuple[np.ndarray, ...]]:
        assert len(self) == len(permutation)
        for i in range(0, len(self), batch_size):
            batch_idxs = permutation[i:i + batch_size]
            if len(batch_idxs) == 0:
                raise StopIteration

            # Sentence-level data
            sentence_tensors, sentence_dict = NLPData._create_sentence_tensors(self.sentence_ids[batch_idxs])
            # Word-level data
            word_tensors, word_dict = self._create_word_tensors(sentence_dict, padding=padding)
            # Char-level data
            char_tensors = self._create_char_tensors(word_dict, padding=padding)
            yield (*sentence_tensors, *word_tensors, *char_tensors)


class StoriesDataset:

    def __init__(self,
                 df: pd.DataFrame,
                 SENTENCES: int = 4,
                 ENDINGS: int = 2,
                 label_dictionary: Dict[int, int] = {
                         1: 0,
                         2: 1
                 },
                 balanced_batches: bool = False) -> None:
        self._len: int = len(df)
        if 'storyid' in df.columns:
            self.story_ids: Sequence[str] = df['storyid'].values
        self.label_dictionary: Dict[int, int] = label_dictionary
        self.balanced_batches = balanced_batches
        if 'label' in df.columns:
            self.labels: Sequence[int] = np.fromiter(
                    (self.label_dictionary[label] for label in df['label'].values), dtype=np.int32)

        self.SENTENCES = SENTENCES
        self.ENDINGS = ENDINGS
        self.TOTAL_SENT = SENTENCES + ENDINGS

    def __len__(self) -> int:
        return self._len

    @staticmethod
    def _sequence_batch_iterator(seq: Sequence[T], permutation: Sequence[int],
                                 batch_size: int = 1) -> Iterator[Sequence[T]]:
        for i in range(0, len(seq), batch_size):
            batch = list()
            for idx in permutation[i:i + batch_size]:
                batch.append(seq[idx])
            yield batch

    def batch_per_epoch_generator(self, batch_size: int = 1,
                                  shuffle: bool = True) -> Iterator[Dict[str, Union[np.ndarray, bool]]]:
        """To be overridden."""
        pass

    def batches_per_epoch(self, batch_size: int = 1, shuffle: bool = True):
        n_batches = self.n_batches(batch_size=batch_size)
        batch_generator = self.batch_per_epoch_generator(batch_size=batch_size, shuffle=shuffle)
        return n_batches, batch_generator

    def n_batches(self, batch_size: int = 1):
        remainder = 0 if len(self) % batch_size == 0 else 1
        return len(self) // batch_size + remainder


class NLPStoriesDataset(StoriesDataset):

    def __init__(self, df: pd.DataFrame, skip_thought_folder: str, *args, vocabularies: Vocabularies = None,
                 **kwargs) -> None:
        super().__init__(df, *args, **kwargs)
        dataset: List[DatasetRow] = self._create_nlp_text_dataset(df)
        del df

        if vocabularies is not None:
            self.vocabularies = vocabularies
        else:
            self.vocabularies = Vocabularies(dataset, skip_thought_folder=skip_thought_folder)

        self.sentences: NLPData = NLPData(dataset, vocabularies=self.vocabularies)

    def _create_nlp_text_dataset(self, df: pd.DataFrame) -> List[DatasetRow]:
        # Index into Pandas DataFrame
        sentence_indexer = create_sentence_indexer(n_sentences=self.SENTENCES, n_endings=self.ENDINGS)
        rows: List[DatasetRow] = []
        for i in range(len(df)):
            row: List[Sentence] = []
            for key in sentence_indexer:
                sentence: str = df[key].values[i]
                sentence_split: Sentence = sentence.strip().split()
                row.append(sentence_split)
            row_tuple: DatasetRow = tuple(row)
            assert len(row) == self.TOTAL_SENT
            rows.append(row_tuple)
        return rows

    def batch_per_epoch_generator(self, batch_size: int = 1,
                                  shuffle: bool = True) -> Iterator[Dict[str, Union[np.ndarray, bool]]]:
        if self.balanced_batches and hasattr(self, 'labels'):
            permutation = generate_balanced_permutation(self.labels, batch_size=batch_size, shuffle=shuffle)
        else:
            permutation = generate_permutation(len(self), shuffle=shuffle)
        sentences_it = self.sentences.batch_iterator(permutation, batch_size=batch_size)
        if hasattr(self, 'labels'):
            labels_it = StoriesDataset._sequence_batch_iterator(self.labels, permutation, batch_size=batch_size)
        for n_batch in range(self.n_batches(batch_size=batch_size)):
            batch_to_sentence_ids, batch_to_sentences, sentence_to_word_ids, sentence_to_words, sentence_lens, \
                word_to_char_ids, word_to_chars, word_lens = next(sentences_it)
            assert batch_to_sentence_ids.shape[0] <= batch_size
            assert batch_to_sentences.shape == batch_to_sentence_ids.shape
            feed_dict = {
                    "batch_to_sentence_ids": batch_to_sentence_ids,
                    "batch_to_sentences": batch_to_sentences,
                    "sentence_to_word_ids": sentence_to_word_ids,
                    "sentence_to_words": sentence_to_words,
                    "sentence_lens": sentence_lens,
                    "word_to_char_ids": word_to_char_ids,
                    "word_to_chars": word_to_chars,
                    "word_lens": word_lens,
                    "is_training": shuffle,
            }
            if hasattr(self, 'labels'):
                feed_dict["labels"] = next(labels_it)
            yield feed_dict


class SkipThoughtStoriesDataset(StoriesDataset):

    def __init__(self, df: pd.DataFrame, encoder: EncoderManager, *args, add: Optional[str] = None, **kwargs) -> None:
        super().__init__(df, *args, **kwargs)
        self.encoder = encoder
        sentence_idx = create_sentence_indexer(self.SENTENCES, self.ENDINGS)

        print("Encoding sentence embeddings...", flush=True)
        self.mat = np.zeros((len(df), self.TOTAL_SENT), dtype=np.int32)
        s_vocab: Dict[str, int] = OrderedDict()
        m = df[sentence_idx].values

        for i, row in enumerate(m):
            for j, sentence in enumerate(row):
                if sentence not in s_vocab:
                    s_vocab[sentence] = len(s_vocab)
                self.mat[i, j] = s_vocab[sentence]
        del m
        del df

        self.add = np.load(add) if add else None
        self.embs = np.zeros((len(s_vocab), 4800), dtype=np.float32)
        chunk_size = 500
        keys = list(s_vocab.keys())
        for i in range(0, len(self.embs), chunk_size):
            self.embs[i:i + chunk_size, :] = self._encode(keys[i:i + chunk_size])

    def _encode(self, column: Sequence[str]) -> np.ndarray:
        res = self.encoder.encode(column)
        return res

    def batch_iterator(self, permutation: Sequence[int], batch_size: int = 1) -> Iterator[np.ndarray]:
        if hasattr(self, 'labels') and self.add is not None:
            labels_it = StoriesDataset._sequence_batch_iterator(self.labels, permutation, batch_size=batch_size)
        for i in range(0, len(self), batch_size):
            selection = self.mat[permutation[i:i + batch_size]]
            batch = self.embs[selection]
            if hasattr(self, 'labels') and self.add is not None:
                labels = next(labels_it)
                batch[labels == 0, 4, :] += self.add
            yield batch

    def batch_per_epoch_generator(self, batch_size: int = 1,
                                  shuffle: bool = True) -> Iterator[Dict[str, Union[np.ndarray, bool]]]:
        if self.balanced_batches and hasattr(self, 'labels'):
            permutation = generate_balanced_permutation(self.labels, batch_size=batch_size, shuffle=shuffle)
        else:
            permutation = generate_permutation(len(self), shuffle=shuffle)

        if hasattr(self, 'labels'):
            labels_it = StoriesDataset._sequence_batch_iterator(self.labels, permutation, batch_size=batch_size)
        sentences_it = self.batch_iterator(permutation, batch_size=batch_size)
        for n_batch in range(self.n_batches(batch_size=batch_size)):
            batch = next(sentences_it)
            assert batch.shape[0] <= batch_size
            feed_dict = {"batch": batch, "is_training": shuffle}
            if hasattr(self, 'labels'):
                feed_dict["labels"] = next(labels_it)
            yield feed_dict
