import collections
from typing import *

from dotmap import DotMap
import numpy as np
import pandas as pd

from .utils import map_all, invert_vocab, MissingDict, UNK_TOKEN, BASE_VOCAB


class Batch:
    pass


class StoriesDataset:
    """Class capable of loading the stories dataset."""

    def __init__(self,
                 sentences: List[str],
                 endings: List[str],
                 labels: List[int],
                 word_vocab: Dict[str, int] = None,
                 train: 'StoriesDataset' = None) -> None:
        """
        Load dataset from the given files.

        Arguments:
        train: If given, the vocabularies from the training data will be reused.
        """
        self.is_train = train is None  # if train is none, this dataset is the training one

        # Create vocabulary_maps
        if train:
            self._vocabulary_maps = train._vocabulary_maps
        else:
            self._vocabulary_maps = {'chars': {'$pad$': 0, '$unk$': 1}, 'labels': {1: 0, 2: 1}}
            if word_vocab:
                self._vocabulary_maps['words'] = word_vocab
            else:
                self._vocabulary_maps['words'] = {0: 0, 1: 1},  # pad = 0, unk = 1

        self._word_ids = []
        self._charseq_ids = []
        self._charseqs_map = {'$pad$': 0}
        self._charseqs = []
        if labels:
            self._labels = []

        # Load the sentences
        for idx, sentence_row in enumerate(sentences):
            if labels:  # if not test
                label = labels[idx]
                assert label in self._vocabulary_maps['labels']
                self._labels.append(self._vocabulary_maps['labels'][label])

            lines = sentence_row + endings[idx]
            for idx, line in lines:
                self._word_ids.append([])
                self._charseq_ids.append([])
                for word in line:
                    # Characters
                    if word not in self._charseqs_map:
                        self._charseqs_map[word] = len(self._charseqs)
                        self._charseqs.append([])
                        for c in word:
                            if c not in self._vocabulary_maps['chars']:
                                if not train:
                                    self._vocabulary_maps['chars'][c] = len(self._vocabulary_maps['chars'])
                                else:
                                    c = '$unk$'
                            self._charseqs[-1].append(self._vocabulary_maps['chars'][c])
                    self._charseq_ids[-1].append(self._charseqs_map[word])

                    # Words -- MissingDict handles UNKs automatically
                    self._word_ids[-1].append(self._vocabulary_maps['words'][word])

            # Compute sentence lengths
            sentences = len(self._word_ids)
            self._sentence_lens = np.zeros([sentences], np.int32)
            for i in range(sentences):
                self._sentence_lens[i] = len(self._word_ids[i])

            # Create vocabularies
            if train:
                self._vocabularies = train._vocabularies
            else:
                self._vocabularies = {}
                for feature, words in self._vocabulary_maps.items():
                    self._vocabularies[feature] = [""] * len(words)
                    for word, id in words.items():
                        self._vocabularies[feature][id] = word

    def next_batch(self, batch_size: int) -> Tuple:
        """
        Return the next batch.

        Arguments:
        Returns: (sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, labels)
        sequence_lens: batch of sentence_lens
        word_ids: batch of word_ids
        charseq_ids: batch of charseq_ids (the same shape as word_ids, but with the ids pointing into charseqs).
        charseqs: unique charseqs in the batch, indexable by charseq_ids;
          contain indices of characters from vocabulary('chars')
        charseq_lens: length of charseqs
        labels: batch of labels

        batch: [string]

        sequence_lens: tweet -> len([word_id]) == len([charseq_id]) # number of words per tweet
        word_ids: tweet -> [word_id] #
        word_vocab: word -> word_id
        charseq_ids: tweet -> [charseq_id]
        charseqs: charseq_id -> [char_id]
        charseq_lens: word_id -> len([char_id])
        char_vocab: char -> char_id
        """

        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        return self._next_batch(batch_perm)

    def _next_batch(self, batch_perm):
        batch_size = len(batch_perm)

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        batch_word_ids = np.zeros([batch_size, max_sentence_len], np.int32)
        for i in range(batch_size):
            batch_word_ids[i, 0:batch_sentence_lens[i]] = self._word_ids[batch_perm[i]]

        if hasattr(self, '_labels'):  # not test
            batch_labels = np.zeros([batch_size], np.int32)
            for i in range(batch_size):
                batch_labels[i] = self._labels[batch_perm[i]]
        else:
            batch_labels = None

        # Character-level data
        batch_charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
        charseqs_map, charseqs, charseq_lens = {}, [], []
        for i in range(batch_size):
            for j, charseq_id in enumerate(self._charseq_ids[batch_perm[i]]):
                if charseq_id not in charseqs_map:
                    charseqs_map[charseq_id] = len(charseqs)
                    charseqs.append(self._charseqs[charseq_id])
                batch_charseq_ids[i, j] = charseqs_map[charseq_id]

        batch_charseq_lens = np.array([len(charseq) for charseq in charseqs], np.int32)
        batch_charseqs = np.zeros([len(charseqs), np.max(batch_charseq_lens)], np.int32)
        for i in range(len(charseqs)):
            batch_charseqs[i, 0:len(charseqs[i])] = charseqs[i]

        return batch_sentence_lens, batch_word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens, batch_labels


T = TypeVar('T')


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


Word = TypeVar('Seq', str)
Sentence = TypeAlias(Sequence[Word])
IntSentence = TypeAlias(Sequence[int])
Sentences = TypeAlias(Sequence[Sentence])


class IntSentences(Sequence[IntSentence]):
    sentences: Sequence[IntSentence] = None

    def __init__(self, sentences: Sequence[Sentence], vocabulary: Dict[Word, int]):
        self.sentences = map_all(sentences, vocabulary)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, item) -> IntSentence:
        return self.sentences[item]

    def batch_iterator(self, permutation: Sequence[int], batch_size: int = 1,
                       padding: int = 0) -> Generator[Tuple[np.ndarray, np.ndarray]]:
        assert len(self) == len(permutation)
        for i in range(0, len(self), batch_size):
            batch_idxs = permutation[i:i + batch_size]

            if len(batch_idxs) == 0:
                raise StopIteration

            max_seq_len = max((len(seq) for seq in self.sentences))

            seqs = np.full((len(batch_idxs), max_seq_len), fill_value=padding, dtype=np.int32)
            lens = np.zeros(len(batch_idxs), dtype=np.int32)
            for j, idx in enumerate(batch_idxs):
                seq = self.sentences[idx]
                lens[j] = len(seq)
                seqs[j, :lens[j]] = np.asarray(seq)

            yield (seqs, lens)


class IntSentencesFrame(Sequence[IntSentences]):
    """
    Assumes all sequences have same length!!!
    """

    int_sentences_frame: Sequence[IntSentences] = None
    vocabulary: Dict[Word, int] = None
    inverse_vocabulary: Dict[int, Word] = None

    def __init__(self, sentences_frame: Sequence[Sentences], vocabulary: Dict[Word, int] = None):
        assert len(sentences_frame) > 0
        for i in range(1, len(sentences_frame)):
            assert len(sentences_frame[i - 1]) == len(sentences_frame[i])

        if vocabulary is None:
            self.vocabulary = IntSentencesFrame._build_vocabulary(sentences_frame)
        else:
            self.vocabulary = vocabulary
        self.inverse_vocabulary = invert_vocab(self.vocabulary)

        self.int_sentences_frame = [IntSentences(sentences, self.vocabulary) for sentences in sentences_frame]

    @staticmethod
    def _build_vocabulary(columns: Sequence[Sentences]) -> Dict[Word, int]:
        vocab = MissingDict(BASE_VOCAB, default_val=BASE_VOCAB[UNK_TOKEN])
        for column in columns:
            for sentence in column:
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = len(vocab)
        return vocab

    def batch_iterator(self, permutation, batch_size=1, padding=0):
        iterators = [seq.batch_iterator(permutation, batch_size=batch_size) for seq in self.int_sentences_frame]
        for i in range(0, len(self.int_sentences_frame[0]), batch_size):
            batches = [next(it) for it in iterators]
            max_batch_len = max(batch.shape[1] for batch, lens in batches)
            big_batch = np.full((batch_size, len(iterators), max_batch_len), fill_value=padding)
            big_lens = np.zeros((batch_size, len(iterators)))
            for j, (batch, lens) in enumerate(batches):
                big_batch[:, j, :batch.shape[1]] = batch
                big_lens[:, j] = lens
            yield (big_batch, big_lens)

    def __len__(self):
        return len(self.int_sentences_frame)

    def __getitem__(self, item):
        return self.int_sentences_frame[item]


class StoriesDataset2:
    SENTENCES = 4
    ENDINGS = 2
    TOTAL_SENT = SENTENCES + ENDINGS

    story_ids: Sequence[str] = None
    word_ids: IntSentencesFrame = None
    charseq_ids: IntSentencesFrame = None
    # charseqs: Sequence[Word] = None
    labels: Sequence[int] = None
    _len: int = None

    def __init__(self, df: pd.DataFrame, data_train: 'StoriesDataset2' = None) -> None:
        self._len = len(df)
        self.story_ids = df['storyid'].values
        self.labels = df['labels'].values

        sentence_indexer = [f"sentence{i+1}" for i in range(self.SENTENCES)]\
            + [f"ending{i+1}" for i in range(self.ENDINGS)]

        word_sentences_seqs = [[sentence.strip().split() for sentence in df[key].values] for key in sentence_indexer]
        charseq_sentences_seqs = [df[key].values for key in sentence_indexer]
        self.word_ids = IntSentencesFrame(word_sentences_seqs, vocabulary=data_train.word_ids.vocabulary)
        self.charseq_ids = IntSentencesFrame(charseq_sentences_seqs, vocabulary=data_train.charseq_ids.vocabulary)
        # TODO: Charseqs

    def __len__(self) -> int:
        return self._len

    @staticmethod
    def _sequence_batch_iterator(seq: Sequence[T], permutation: Sequence[int],
                                 batch_size: int = 1) -> Generator[Sequence[T]]:
        for i in range(0, len(seq), batch_size):
            batch = list()
            for idx in permutation[i:i + batch_size]:
                batch.append(seq[idx])
            yield batch

    def batch_per_epoch_generator(self, batch_size: int = 1, shuffle: bool = True) -> Generator[Batch]:
        permutation = generate_balanced_permutation(self.labels, batch_size=batch_size, shuffle=shuffle)
        word_ids_it = self.word_ids.batch_iterator(permutation, batch_size=batch_size)
        charseq_ids_it = self.charseq_ids.batch_iterator(permutation, batch_size=batch_size)
        labels_it = StoriesDataset2._sequence_batch_iterator(self.labels, permutation, batch_size=batch_size)
        for n_batch in range(self.n_batches(batch_size=batch_size)):
            word_ids, word_lens = next(word_ids_it)
            charseq_ids, charseq_lens = next(charseq_ids_it)
            assert word_lens.shape == charseq_lens.shape
            assert word_lens == charseq_lens
            labels = next(labels_it)
            yield DotMap({
                    "sentence_word_ids": word_ids[:, :self.SENTENCES, :],
                    "sentence_charseq_ids": charseq_ids[:, :self.SENTENCES, :],
                    "ending_word_ids": word_ids[:, -self.ENDINGS:, :],
                    "ending_charseq_ids": charseq_ids[:, -self.ENDINGS:, :],
                    "word_lens": word_lens[:, :self.SENTENCES],
                    "ending_lens": word_lens[:, -self.ENDINGS:],
                    "charseqs": None,
                    "charseq_lens": None,
                    "labels": labels,
                    "is_training": shuffle
            })

    def batches_per_epoch(self, batch_size: int = 1, shuffle: bool = True):
        n_batches = self.n_batches(batch_size=batch_size)
        batch_generator = self.batch_per_epoch_generator(batch_size=batch_size, shuffle=shuffle)
        return n_batches, batch_generator

    def n_batches(self, batch_size: int = 1):
        return len(self) // batch_size + len(self) % batch_size
