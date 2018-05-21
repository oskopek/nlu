from typing import Tuple

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
                 grad_clip: float = 10.0,
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
        self.grad_clip = grad_clip

    def _create_cell(self) -> tf.nn.rnn_cell.LayerRNNCell:
        if self.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(self.rnn_cell_dim)
        elif self.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(self.rnn_cell_dim)
        else:
            raise ValueError("Unknown rnn_cell {}".format(self.rnn_cell))

    def _char_embeddings(self) -> tf.Tensor:
        if self.char_embedding == -1:
            input_chars = tf.one_hot(self.word_to_chars, depth=self.num_chars)
        else:
            char_emb_mat = tf.get_variable("char_emb", shape=[self.num_chars, self.char_embedding])
            input_chars = tf.nn.embedding_lookup(char_emb_mat, self.word_to_chars)
        print("input_chars", input_chars.get_shape())

        rnn_cell_characters = self._create_cell()
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell_characters,
                cell_bw=rnn_cell_characters,
                inputs=input_chars,
                sequence_length=self.word_lens,
                dtype=tf.float32,
                scope="rnn_chars")
        input_chars = tf.concat([state_fw, state_bw], axis=1)
        print("input_chars_rnn", input_chars.get_shape())

        input_char_words = tf.nn.embedding_lookup(input_chars, self.sentence_to_words)
        print("input_char_words", input_char_words.get_shape())
        return input_char_words

    def _word_embeddings(self) -> tf.Tensor:
        # [batch_size, SENTENCE x sentence_id, max_word_ids]
        batch_to_sentences = tf.nn.embedding_lookup(self.sentence_to_words, ids=self.batch_to_sentences)
        shape = batch_to_sentences.get_shape()
        print("batch_to_sentences", shape)
        sentence_to_words = tf.reshape(batch_to_sentences, tf.stack([shape[0] * shape[1], shape[2], shape[3]]))

        if self.word_embedding == -1:
            sentence_word_embeddings = tf.one_hot(sentence_to_words, self.num_words)
        else:
            word_emb_mat = tf.get_variable("word_emb", shape=[self.num_words, self.word_embedding])
            sentence_word_embeddings = tf.nn.embedding_lookup(word_emb_mat, sentence_to_words)
            sentence_word_embeddings = tf.layers.dropout(
                    sentence_word_embeddings, rate=self.keep_prob, training=self.is_training)
        print("sentence_word_embeddings", sentence_word_embeddings.get_shape())
        return sentence_word_embeddings

    def _sentence_rnn(self, inputs: tf.Tensor) -> tf.Tensor:
        lens = tf.nn.embedding_lookup(self.sentence_lens, self.batch_to_sentences)
        lens_flat = tf.reshape(lens, (-1))

        rnn_cell_words = self._create_cell()
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_bw=rnn_cell_words,
                cell_fw=rnn_cell_words,
                inputs=inputs,
                sequence_length=lens_flat,
                dtype=tf.float32)
        sentence_wordword_states = tf.concat([state_fw, state_bw], axis=1)
        print("sentence_wordword_states", sentence_wordword_states.get_shape())
        return sentence_wordword_states

    def _story_embeddings(self, sentence_wordword_states: tf.Tensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        states_unflat = tf.reshape(sentence_wordword_states, (-1, self.SENTENCES + self.ENDINGS, self.rnn_cell_dim * 2))
        print("states_unflat", states_unflat.get_shape())
        states_sentences = states_unflat[:, :self.SENTENCES, :]
        states_endings = states_unflat[:, -self.ENDINGS:, :]
        print("states separate", states_sentences.get_shape(), states_endings.get_shape())

        sentence_state = tf.layers.flatten(states_sentences)
        ending1_state = tf.squeeze(states_endings[:, 0, :], axis=1)
        ending2_state = tf.squeeze(states_endings[:, 1, :], axis=1)
        ending1_state = tf.layers.flatten(ending1_state)
        ending2_state = tf.layers.flatten(ending2_state)
        return sentence_state, (ending1_state, ending2_state)

    def _fc(self, sentence_state: tf.Tensor, ending_states: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        with tf.variable_scope("sentences_fc"):
            sentences_fc = tf.layers.dense(sentence_state, 1024, activation=tf.nn.leaky_relu)
            sentences_fc = tf.layers.dense(sentences_fc, 512, activation=tf.nn.leaky_relu)
            print("sentences_fc", sentences_fc.get_shape())
        with tf.variable_scope("endings_fc"):
            ending_fc = tf.layers.dense(ending_states[0], 1024, activation=tf.nn.leaky_relu, name="fc1")
            ending1_output = tf.layers.dense(ending_fc, 512, activation=tf.nn.leaky_relu, name="fc2")
            print("ending1_output", ending1_output.get_shape())
        with tf.variable_scope("endings_fc", reuse=tf.AUTO_REUSE):
            ending_fc = tf.layers.dense(ending_states[1], 1024, activation=tf.nn.leaky_relu, name="fc1")
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
            with tf.name_scope("charword_embeddings"):
                sentence_charword_states = self._char_embeddings()

            with tf.name_scope("word_embeddings"):
                sentence_word_embeddings = self._word_embeddings()

            with tf.name_scope("sentence_embeddings"):
                inputs = tf.concat([sentence_charword_states, sentence_word_embeddings], axis=2)
                sentence_wordword_states = self._sentence_rnn(inputs)

            with tf.name_scope("story_embeddings"):
                states_sentences, states_endings = self._story_embeddings(sentence_wordword_states)

            with tf.name_scope("fc"):
                output_layer = self._fc(states_sentences, states_endings)

            predictions = tf.cast(tf.argmax(output_layer, 1), tf.int32, name="predictions")

            with tf.name_scope("loss"):
                loss = tf.losses.sparse_softmax_cross_entropy(
                        self.labels, output_layer, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gradients = optimizer.compute_gradients(self.loss)
                clipped_gradients = [(tf.clip_by_norm(gradient, self.grad_clip), var) for gradient, var in gradients]
                training_step = optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)

        return predictions, loss, training_step
