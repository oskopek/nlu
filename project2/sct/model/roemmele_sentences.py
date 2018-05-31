from typing import Tuple, Dict, Union

from .rnn import RNN

import numpy as np
import tensorflow as tf


class RoemmeleSentences(RNN):
    CLASSES = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

    def _sentence_states(self) -> tf.Tensor:
        return self.batch

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
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                gradients = optimizer.compute_gradients(loss)
                clipped_gradients = [(tf.clip_by_norm(gradient, self.grad_clip), var) for gradient, var in gradients]
                training_step = optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)

                variables = tf.trainable_variables()
                print("Variables", variables)

        return eval_predictions, loss, training_step

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
        self.saver = tf.train.Saver(max_to_keep=4)

        # Initialize variables
        self.session.run(tf.global_variables_initializer())
        with self.summary_writer.as_default():
            tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def _placeholders(self) -> None:
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

        # [batch_size, SENTENCES x sentence_id, sentence_embeddings]
        self.batch = tf.placeholder(tf.float32, [None, self.TOTAL_SENTENCES, self.sentence_embedding], name="batch")

        # [batch_size]
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        # [] bool scalar
        self.is_training = tf.placeholder_with_default(False, [], name="is_training")

        # Useful tensors
        self.batch_size = tf.shape(self.batch)[0]

    def _build_feed_dict(self, batch: Dict[str, Union[np.ndarray, bool]],
                         is_training: bool = False) -> Dict[tf.Tensor, Union[np.ndarray, bool]]:
        assert is_training == batch['is_training']
        return {self.batch: batch['batch'], self.labels: batch['labels'], self.is_training: batch['is_training']}
