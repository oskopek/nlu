from typing import *

import tensorflow as tf

from .model import Model


class RNN(Model):
    CLASSES = 2

    def __init__(self,
                 rnn_cell: str,
                 rnn_cell_dim: int,
                 num_words: int,
                 num_chars: int,
                 *args,
                 word_embedding: int = 100,
                 char_embedding: int = 100,
                 keep_prob: float = 0.5,
                 learning_rate: float = 1e-4,
                 **kwargs) -> None:
        super(RNN, self).__init__(*args, **kwargs)
        self.rnn_cell = rnn_cell
        self.rnn_cell_dim = rnn_cell_dim
        self.keep_prob = keep_prob
        self.num_words = num_words
        self.num_chars = num_chars
        self.char_embedding = char_embedding
        self.word_embedding = word_embedding
        self.learning_rate = learning_rate

    def _create_cell(self) -> tf.nn.rnn_cell.LayerRNNCell:
        if self.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(self.rnn_cell_dim)
        elif self.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(self.rnn_cell_dim)
        else:
            raise ValueError("Unknown rnn_cell {}".format(self.rnn_cell))

    def _char_embeddings(self) -> tf.Tensor:
        if self.char_embedding == -1:
            input_chars = tf.one_hot(self.charseqs, self.num_chars)
        else:
            char_emb_mat = tf.get_variable("char_emb", shape=[self.num_chars, self.char_embedding])
            input_chars = tf.nn.embedding_lookup(char_emb_mat, self.charseqs)
        print("input_chars", input_chars.get_shape())
        return input_chars

    def _char_rnn(self, input_chars: tf.Tensor) -> tf.Tensor:
        rnn_cell_characters = self._create_cell()
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_characters,
                cell_bw=rnn_cell_characters,
                inputs=input_chars,
                sequence_length=self.charseq_lens,
                dtype=tf.float32,
                scope="rnn_chars")
        input_chars = tf.concat([state_fw, state_bw], axis=1)
        print("input_chars_rnn", input_chars.get_shape())

        charseq_ids = tf.concat([self.sentence_charseq_ids, self.ending_charseq_ids], axis=1)
        print("charseq_ids", charseq_ids.get_shape())
        shape = charseq_ids.get_shape()
        charseq_ids_flat = tf.reshape(charseq_ids, tf.stack([shape[0] * shape[1], shape[2], shape[3]]))
        input_char_words = tf.nn.embedding_lookup(input_chars, charseq_ids_flat)
        input_char_words = tf.layers.dropout(input_char_words, rate=self.keep_prob, training=self.is_training)
        print("input_char_words", input_char_words.get_shape())
        return input_char_words

    def _word_embeddings(self) -> tf.Tensor:
        word_ids = tf.concat([self.sentence_word_ids, self.ending_word_ids], axis=1)
        print("word_ids", word_ids.get_shape())
        shape = word_ids.get_shape()
        word_ids_flat = tf.reshape(word_ids, tf.stack([shape[0] * shape[1], shape[2], shape[3]]))

        if self.word_embedding == -1:
            input_words = tf.one_hot(word_ids_flat, self.num_words)
        else:
            word_emb_mat = tf.get_variable("word_emb", shape=[self.num_words, self.word_embedding])
            input_words = tf.nn.embedding_lookup(word_emb_mat, word_ids_flat)
            input_words = tf.layers.dropout(input_words, rate=self.keep_prob, training=self.is_training)
        print("input_words", input_words.get_shape())
        return input_words

    def _word_rnn(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        lens = tf.concat([self.sentence_lens, self.ending_lens], axis=1)
        lens_flat = tf.reshape(lens, (-1))

        rnn_cell_words = self._create_cell()
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_bw=rnn_cell_words,
                cell_fw=rnn_cell_words,
                inputs=inputs,
                sequence_length=lens_flat,
                dtype=tf.float32)
        states = tf.concat([state_fw, state_bw], axis=1)
        print("states", states.get_shape())
        states_unflat = tf.reshape(states, (-1, self.SENTENCES, self.ENDINGS, self.rnn_cell_dim * 2))
        print("states_unflat", states_unflat.get_shape())
        states_sentences = states_unflat[:, :self.SENTENCES, :]
        states_endings = states_unflat[:, -self.ENDINGS:, :]
        print("states separate", states_sentences.get_shape(), states_endings.get_shape())
        return states_sentences, states_endings

    def _fc(self, states_sentences: tf.Tensor, states_endings: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("sentences_fc"):
            sentences_flat = tf.layers.flatten(states_sentences)
            sentences_fc = tf.layers.dense(sentences_flat, 1024, activation=tf.nn.leaky_relu)
            sentences_fc = tf.layers.dense(sentences_fc, 512, activation=tf.nn.leaky_relu)
            print("sentences_fc", sentences_fc.get_shape())
        with tf.variable_scope("endings_fc"):
            ending = tf.squeeze(states_endings[:, 0, :], axis=1)
            ending_flat = tf.layers.flatten(ending)
            ending_fc = tf.layers.dense(ending_flat, 1024, activation=tf.nn.leaky_relu, name="fc1")
            ending1_output = tf.layers.dense(ending_fc, 512, activation=tf.nn.leaky_relu, name="fc2")
            print("ending1_output", ending1_output.get_shape())
        with tf.variable_scope("endings_fc"):
            ending = tf.squeeze(states_endings[:, 1, :], axis=1)
            ending_flat = tf.layers.flatten(ending)
            ending_fc = tf.layers.dense(ending_flat, 1024, activation=tf.nn.leaky_relu, name="fc1")
            ending2_output = tf.layers.dense(ending_fc, 512, activation=tf.nn.leaky_relu, name="fc2")
            print("ending2_output", ending2_output.get_shape())
        with tf.variable_scope("common_fc"):
            flatten = tf.concat([sentences_fc, ending1_output, ending2_output], axis=1)
            fc = tf.layers.dense(flatten, 1024, activation=tf.nn.leaky_relu, name="fc1")
            fc = tf.layers.dense(fc, 512, activation=tf.nn.leaky_relu, name="fc2")
            output = tf.layers.dense(fc, self.CLASSES, activation=None, name="output")
            print("output", output.get_shape())
        return output

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        # Construct the graph
        with self.session.graph.as_default():
            with tf.name_scope("char_embeddings"):
                input_chars = self._char_embeddings()

            with tf.name_scope("char_rnn"):
                input_char_words = self._char_rnn(input_chars)

            with tf.name_scope("word_embeddings"):
                input_words = self._word_embeddings()

            inputs = tf.concat([input_char_words, input_words], axis=2)
            print("inputs", inputs.get_shape())

            with tf.name_scope("word_rnn"):
                states_sentences, states_endings = self._word_rnn(inputs)

            with tf.name_scope("fc"):
                output_layer = self._fc(states_sentences, states_endings)

            predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32, name="predictions")

            with tf.name_scope("loss"):
                loss = tf.losses.sparse_softmax_cross_entropy(
                        self.labels, output_layer, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gradients = optimizer.compute_gradients(self.loss)
                clipped_gradients = [(tf.clip_by_norm(gradient, 5), var) for gradient, var in gradients]
                training_step = optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)

        return predictions, loss, training_step
