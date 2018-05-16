from .model import Model
import tensorflow as tf


class RNN(Model):
    CLASSES = 2

    def __init__(self,
                 rnn_cell,
                 rnn_cell_dim,
                 num_words,
                 num_chars,
                 *args,
                 word_embedding=100,
                 char_embedding=100,
                 keep_prob=0.5,
                 learning_rate=1e-4,
                 **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.rnn_cell = rnn_cell
        self.rnn_cell_dim = rnn_cell_dim
        self.keep_prob = keep_prob
        self.num_words = num_words
        self.num_chars = num_chars
        self.char_embedding = char_embedding
        self.word_embedding = word_embedding
        self.learning_rate = learning_rate

    def _create_cell(self):
        if self.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(self.rnn_cell_dim)
        elif self.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(self.rnn_cell_dim)
        else:
            raise ValueError("Unknown rnn_cell {}".format(self.rnn_cell))

    def _char_embeddings(self):
        if self.char_embedding == -1:
            input_chars = tf.one_hot(self.charseqs, self.num_chars)
        else:
            char_emb_mat = tf.get_variable("char_emb", shape=[self.num_chars, self.char_embedding])
            input_chars = tf.nn.embedding_lookup(char_emb_mat, self.charseqs)
        print("input_chars", input_chars.get_shape())
        return input_chars

    def _char_rnn(self, input_chars):
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

        input_char_words = tf.nn.embedding_lookup(input_chars, self.charseq_ids)
        input_char_words = tf.layers.dropout(input_char_words, rate=self.keep_prob, training=self.is_training)
        print("input_char_words", input_char_words.get_shape())
        return input_char_words

    def _word_embeddings(self):
        if self.word_embedding == -1:
            input_words = tf.one_hot(self.word_ids, self.num_words)
        else:
            word_emb_mat = tf.get_variable("word_emb", shape=[self.num_words, self.word_embedding])
            input_words = tf.nn.embedding_lookup(word_emb_mat, self.word_ids)
            input_words = tf.layers.dropout(input_words, rate=self.keep_prob, training=self.is_training)
        print("input_words", input_words.get_shape())
        return input_words

    def _word_rnn(self, inputs):
        rnn_cell_words = self._create_cell()
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_bw=rnn_cell_words,
                cell_fw=rnn_cell_words,
                inputs=inputs,
                sequence_length=self.sentence_lens,
                dtype=tf.float32)
        states = tf.concat([state_fw, state_bw], axis=1)
        print("states", states.get_shape())
        return states

    def _fc(self, states):
        hidden = tf.layers.dense(states, 64, activation=tf.nn.leaky_relu)
        d1 = tf.layers.dropout(hidden, rate=self.keep_prob, training=self.is_training)
        output_layer = tf.layers.dense(d1, self.CLASSES, activation=None)
        print("output_layer", output_layer.get_shape())
        return output_layer

    def build_model(self):
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
                states = self._word_rnn(inputs)

            with tf.name_scope("fc"):
                output_layer = self._fc(states)

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
