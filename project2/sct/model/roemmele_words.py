import tensorflow as tf

from .roemmele_sentences import RoemmeleSentences


class RoemmeleWords(RoemmeleSentences):
    CLASSES = 1

    def __init__(self, *args, **kwargs) -> None:
        # Call super last, because our build_model method probably needs above initialization to happen first
        super().__init__(*args, **kwargs)

    def _word_embeddings(self) -> tf.Tensor:
        # [batch_size, SENTENCES, max_word_ids]
        print("self.sentence_to_word_ids", self.sentence_to_word_ids.get_shape())
        print("self.batch_to_sentences", self.batch_to_sentences.get_shape())
        batch_to_sentences = tf.nn.embedding_lookup(self.sentence_to_word_ids, ids=self.batch_to_sentences)
        print("batch_to_sentences", batch_to_sentences.get_shape())

        if self.word_embedding == -1:
            sentence_word_embeddings = tf.one_hot(batch_to_sentences, self.num_words)
        else:
            self.pretrained_embeddings = tf.placeholder(tf.float32, [self.num_words, self.word_embedding])
            word_emb_mat = tf.get_variable("word_emb", shape=[self.num_words, self.word_embedding], trainable=False)
            self.assign_pretrained_we = word_emb_mat.assign(self.pretrained_embeddings)
            sentence_word_embeddings = tf.nn.embedding_lookup(word_emb_mat, ids=batch_to_sentences)
        print("sentence_word_embeddings", sentence_word_embeddings.get_shape())
        return sentence_word_embeddings

    def _word_rnn(self, sentence_word_embeddings: tf.Tensor) -> tf.Tensor:
        batch_sentence_lens = tf.nn.embedding_lookup(self.sentence_lens, self.batch_to_sentences)
        print("batch_sentence_lens", batch_sentence_lens.get_shape())
        batch_sentence_lens_flat = tf.reshape(batch_sentence_lens, (-1,))

        sentence_word_embeddings_flat = tf.reshape(sentence_word_embeddings,
                                                   (self.batch_size * self.TOTAL_SENTENCES, -1, self.word_embedding))
        print("sentence_word_embeddings_flat", sentence_word_embeddings_flat.get_shape())

        # Create the cell
        rnn_cell_dim = self.sentence_embedding
        if self.rnn_cell == "LSTM":
            rnn_cell_dim //= 2
        rnn_cell_words = self._create_cell(rnn_cell_dim)

        _, state = tf.nn.dynamic_rnn(
                cell=rnn_cell_words,
                inputs=sentence_word_embeddings_flat,
                sequence_length=batch_sentence_lens_flat,
                dtype=tf.float32)
        if self.rnn_cell == "LSTM":
            state = tf.concat(state, axis=-1)

        state = tf.reshape(state, (-1, self.TOTAL_SENTENCES, self.sentence_embedding))
        print("per_sentence_states", state.get_shape())
        return state

    def _sentence_states(self) -> tf.Tensor:
        with tf.name_scope("word_embeddings"):
            sentence_word_embeddings = self._word_embeddings()

        with tf.variable_scope("word_rnn"):
            # per_sentence_states = self._word_rnn(sentence_word_embeddings)
            per_sentence_states = tf.reduce_mean(sentence_word_embeddings, axis=2)
        return per_sentence_states
