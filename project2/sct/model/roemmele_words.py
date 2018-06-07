from typing import Tuple

import tensorflow as tf

from .rnn import RNN


# TODO: De-duplicate with RoemmeleSentences?
class RoemmeleWords(RNN):
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

    def _sentence_rnn(self, per_sentence_states: tf.Tensor) -> tf.Tensor:
        assert len(per_sentence_states.get_shape()) == 3
        assert per_sentence_states.get_shape()[1] == self.TOTAL_SENTENCES - 1
        # Create the cell
        rnn_cell_sentences = self._create_cell(self.rnn_cell_dim, name='sentence_cell')

        inputs = tf.unstack(per_sentence_states, axis=1)
        outputs, state = tf.nn.static_rnn(cell=rnn_cell_sentences, inputs=inputs, dtype=tf.float32)
        if self.rnn_cell == "LSTM":
            state = state[0]  # c_state

        print("outputs[0]", outputs[0].get_shape())
        outputs_lst = [tf.expand_dims(x, axis=1) for x in outputs]
        outputs_tensor = tf.concat(outputs_lst, axis=1)
        print("outputs_tensor", outputs_tensor.get_shape())

        sentence_states = [state]
        if self.attention is not None:  # with attention
            with tf.variable_scope("attention"):
                context = self._add_attention(outputs_tensor, cell_output=state, prefix="attention")
                print("context", context.get_shape())
            sentence_states.append(context)

        res = tf.concat(sentence_states, axis=1)
        print("sentence_states", res.get_shape())
        return res

    def _output_fc(self, state: tf.Tensor) -> tf.Tensor:
        output = tf.layers.dense(state, self.CLASSES, activation=None, name="output")
        print("output", output.get_shape())
        return output

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        # Construct the graph
        with self.session.graph.as_default():
            with tf.name_scope("split_endings"):
                per_sentence_states = self._sentence_states()
                sentence_states = per_sentence_states[:, :self.SENTENCES, :]
                print("sentence_states", sentence_states.get_shape())
                ending1_states = per_sentence_states[:, self.SENTENCES + 0, :]
                ending1_states = tf.expand_dims(ending1_states, axis=1)
                print("ending1_states", ending1_states.get_shape())
                ending2_states = per_sentence_states[:, self.SENTENCES + 1, :]
                ending2_states = tf.expand_dims(ending2_states, axis=1)
                print("ending2_states", ending2_states.get_shape())
                ending1_states = tf.concat([sentence_states, ending1_states], axis=1)
                ending2_states = tf.concat([sentence_states, ending2_states], axis=1)

            with tf.variable_scope("ending") as ending_scope:
                with tf.name_scope("sentence_rnn"):
                    per_story_states = self._sentence_module(ending1_states)
                with tf.name_scope("fc"):
                    self.ending1_output = self._output_fc(per_story_states)

            with tf.variable_scope(ending_scope, reuse=True):
                with tf.name_scope("sentence_rnn"):
                    per_story_states = self._sentence_module(ending2_states)
                with tf.name_scope("fc"):
                    self.ending2_output = self._output_fc(per_story_states)

            with tf.name_scope("eval_predictions"):
                endings = tf.concat([self.ending1_output, self.ending2_output], axis=1)
                eval_predictions = tf.to_int32(tf.argmax(endings, axis=1))

            with tf.name_scope("train_predictions"):
                self.train_logits = tf.squeeze(self.ending1_output, axis=[1])
                self.train_probs = tf.sigmoid(self.train_logits)
                self.train_predictions = tf.to_int32(tf.round(self.train_probs))

            with tf.name_scope("loss"):
                loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.labels, logits=self.train_logits)

            with tf.name_scope("optimizer"):
                optimizer = self._optimizer()
                gradients = optimizer.compute_gradients(loss)
                clipped_gradients = [(tf.clip_by_norm(gradient, self.grad_clip), var) for gradient, var in gradients]
                training_step = optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)

                variables = tf.trainable_variables()
                print("Variables", variables)

        return eval_predictions, loss, training_step

    def _optimizer(self) -> tf.train.Optimizer:
        return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

    def _summaries_and_init(self) -> None:
        with tf.name_scope("summaries"):
            current_accuracy, update_accuracy = tf.metrics.accuracy(self.labels, self.train_predictions)
            current_eval_accuracy, update_eval_accuracy = tf.metrics.accuracy(self.labels, self.predictions)
            current_loss, update_loss = tf.metrics.mean(self.loss)
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
            self.current_metrics = [current_accuracy, current_loss]
            self.update_metrics = [update_accuracy, update_loss]
            self.current_eval_metrics = [current_eval_accuracy]

            with self.summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(50):
                self.summaries["train"].extend([
                        tf.contrib.summary.histogram("train/activations", self.train_probs),
                        tf.contrib.summary.scalar("train/loss", update_loss),
                        tf.contrib.summary.scalar("train/accuracy", update_accuracy)
                ])
            with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                eval_histograms = [
                        tf.contrib.summary.histogram("eval/activations1", tf.sigmoid(self.ending1_output)),
                        tf.contrib.summary.histogram("eval/activations2", tf.sigmoid(self.ending2_output))
                ]
                self.update_eval_metrics = [update_eval_accuracy] + eval_histograms
                for dataset in ["eval", "test"]:
                    self.summaries[dataset].append(
                            tf.contrib.summary.scalar(dataset + "/accuracy", current_eval_accuracy))

        # Saver
        self.saver = tf.train.Saver(max_to_keep=3)

        # Initialize variables
        self.session.run(tf.global_variables_initializer())
        with self.summary_writer.as_default():
            tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)
