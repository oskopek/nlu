from typing import Sequence, TypeVar, Tuple, Dict, List, Iterator, Union

from dotmap import DotMap
import numpy as np
import pandas as pd

from .utils import MissingDict, UNK_TOKEN, BASE_VOCAB, invert_dict

T = TypeVar('T')
Char = str
Word = str
Sentence = Sequence[Word]
IntSentence = Sequence[int]
DatasetRow = Tuple[Sentence, ...]
Dataset = Sequence[DatasetRow]


def generate_balanced_permutation(labels: Sequence[T], batch_size: int = 1, shuffle: bool = True) -> Sequence[int]:
    """Currently only works for binary labels and an approximately balanced dataset."""
    if not shuffle:
        return np.arange(len(labels))

    permutation = np.random.permutation(len(labels))  # shuffle all
    label_split = Dict[T, List[int]]()  # split by label
    for i, label in enumerate(labels):
        if label not in label_split:
            label_split[label] = List[int]()
        label_split[label].append(permutation[i])

    balanced_perm = np.zeros_like(permutation)
    one_label, two_label = label_split.keys()
    ones_ratio = len(label_split[one_label]) / len(labels)  # balance batches by ratio
    for n_batch in range(0, len(labels), batch_size):
        counts = Dict[T, int]({k: 0 for k in label_split.keys()})
        for n_sample in range(batch_size):
            cur_ones_ratio = counts[one_label] / sum(counts.values())
            if cur_ones_ratio < ones_ratio:
                chosen_label = one_label
            elif cur_ones_ratio > ones_ratio:
                chosen_label = two_label
            else:
                chosen_label = np.random.choice(label_split.keys(), 1)
            balanced_perm[n_batch * batch_size + n_sample] = label_split[chosen_label].pop()
            counts[chosen_label] += 1
    return balanced_perm


class Vocabularies:

    @staticmethod
    def _default_dict() -> MissingDict[str, int]:
        return MissingDict(BASE_VOCAB, default_val=BASE_VOCAB[UNK_TOKEN])

    def __init__(self, dataset: Sequence[DatasetRow]) -> None:
        self.sentence_vocabulary: Dict[Sentence, int] = MissingDict({[]: 0, [UNK_TOKEN]: 1}, default_val=1)
        self.word_vocabulary: Dict[Word, int] = Vocabularies._default_dict()
        self.char_vocabulary: Dict[Char, int] = Vocabularies._default_dict()

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
            row_sentence_ids = [vocabularies.sentence_vocabulary[sentence] for sentence in row]
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
        batch_to_sentences = np.array(map(lambda idx: sentence_dict[idx], batch_to_sentence_ids), dtype=np.int32)
        return (batch_to_sentence_ids, batch_to_sentences), sentence_dict

    @staticmethod
    def _create_token_tensors(word_ids, sentence_dict: Dict[int, int],
                              padding: int = 0) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[int, int]]:
        inv_sentence_dict = invert_dict(sentence_dict)
        flat_sentence_ids = [inv_sentence_dict[i] for i in range(len(sentence_dict))]
        sentence_lens = np.array((len(word_ids[idx]) for idx in flat_sentence_ids), dtype=np.int32)
        max_sentence_len = np.max(sentence_lens)
        sentence_to_word_ids = np.full((len(flat_sentence_ids), max_sentence_len), fill_value=padding, dtype=np.int32)
        sentence_to_words = np.full((len(flat_sentence_ids), max_sentence_len), fill_value=padding, dtype=np.int32)
        unique_word_ids = list(set(word_id for sent_idx in flat_sentence_ids for word_id in word_ids[sent_idx]))
        word_dict = {k: v for v, k in enumerate(unique_word_ids)}
        for i, idx in enumerate(flat_sentence_ids):
            sentence = word_ids[idx]
            sentence_to_word_ids[i, :sentence_lens[i]] = np.asarray(sentence)
            sentence_to_words[i, :sentence_lens[i]] = np.asarray((word_dict[word_id] for word_id in sentence))
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
    SENTENCES: int = 4
    ENDINGS: int = 2
    TOTAL_SENT: int = SENTENCES + ENDINGS

    def __init__(self, df: pd.DataFrame, vocabularies: Vocabularies = None) -> None:
        self._len: int = len(df)
        self.story_ids: Sequence[str] = df['storyid'].values
        self.labels: Sequence[int] = df['labels'].values

        dataset: List[DatasetRow] = self._create_nlp_text_dataset(df)
        del df

        if vocabularies is None:
            self.vocabularies = Vocabularies(dataset)
        else:
            self.vocabularies = vocabularies

        self.sentences: NLPData = NLPData(dataset, vocabularies=self.vocabularies)

    def __len__(self) -> int:
        return self._len

    def _create_nlp_text_dataset(self, df: pd.DataFrame) -> List[DatasetRow]:
        # Index into Pandas DataFrame
        sentence_indexer = [f"sentence{i+1}" for i in range(self.SENTENCES)]
        sentence_indexer += [f"ending{i+1}" for i in range(self.ENDINGS)]
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
        permutation = generate_balanced_permutation(self.labels, batch_size=batch_size, shuffle=shuffle)
        sentences_it = self.sentences.batch_iterator(permutation, batch_size=batch_size)
        labels_it = StoriesDataset._sequence_batch_iterator(self.labels, permutation, batch_size=batch_size)
        story_ids_it = StoriesDataset._sequence_batch_iterator(self.story_ids, permutation, batch_size=batch_size)
        for n_batch in range(self.n_batches(batch_size=batch_size)):
            batch_to_sentence_ids, batch_to_sentences, sentence_to_word_ids, sentence_to_words, sentence_lens, \
                word_to_char_ids, word_to_chars, word_lens = next(sentences_it)
            assert batch_to_sentence_ids.shape[0] == batch_size
            assert batch_to_sentences.shape == batch_to_sentence_ids.shape
            labels = next(labels_it)
            story_ids = next(story_ids_it)

            yield DotMap({
                    "batch_to_sentence_ids": batch_to_sentence_ids,
                    "batch_to_sentences": batch_to_sentences,
                    "sentence_to_word_ids": sentence_to_word_ids,
                    "sentence_to_words": sentence_to_words,
                    "sentence_lens": sentence_lens,
                    "word_to_char_ids": word_to_char_ids,
                    "word_to_chars": word_to_chars,
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
