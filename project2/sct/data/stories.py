import numpy as np


class StoriesDataset:
    """Class capable of loading the stories dataset."""

    def __init__(self, sentences, endings, labels, word_vocab=None, train=None):
        """Load dataset from the given files.

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

            self._new_permutation()

    def _new_permutation(self):
        if self.is_train:
            self._permutation = np.random.permutation(len(self._sentence_lens))
        else:
            self._permutation = np.arange(len(self._sentence_lens))

    def vocabulary(self, feature):
        """Return vocabulary for required feature.

        The features are the following:
        words
        chars
        labels
        """
        return self._vocabularies[feature]

    def next_batch(self, batch_size):
        """Return the next batch.

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

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False

    def whole_data_as_batch(self):
        """Return the whole dataset in the same result as next_batch.

        Returns the same results as next_batch.
        """
        return self._next_batch(np.arange(len(self._sentence_lens)))

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
