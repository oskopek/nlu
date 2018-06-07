from typing import Tuple, Union, Callable

import tensorflow as tf

from .roemmele_sentences import RoemmeleSentences


class RoemmeleSentencesTCNN(RoemmeleSentences):

    def __init__(self, *args, num_filters: int = 512, **kwargs) -> None:
        self.num_filters = num_filters

        super().__init__(*args, **kwargs)

    @staticmethod
    def _1d_conv(inp: tf.Tensor,
                 num_outputs: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: int = 1,
                 activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
                 normalizer_fn: Callable[[tf.Tensor], tf.Tensor] = tf.layers.batch_normalization):
        res = tf.layers.conv2d(
                inputs=inp,
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                activation=activation_fn,
                padding='valid')
        if normalizer_fn is not None:
            res = normalizer_fn(res)
        return res

    def _sentence_rnn(self, per_sentence_states: tf.Tensor) -> tf.Tensor:
        assert len(per_sentence_states.get_shape()) == 3
        assert per_sentence_states.get_shape()[1] == self.TOTAL_SENTENCES - 1

        expanded = tf.expand_dims(per_sentence_states, axis=-1)
        pooled = []
        for kernel_size in [3, 4, 5]:  # [3, 4, 5, 7]
            c = self._1d_conv(
                    expanded,
                    num_outputs=self.num_filters,
                    kernel_size=(kernel_size, self.sentence_embedding),
                    stride=1)
            print(f"c_{kernel_size}", c.get_shape())
            seq_len = self.SENTENCES + 1
            mp = tf.layers.max_pooling2d(inputs=c, pool_size=[seq_len - kernel_size + 1, 1], strides=1)
            print(f"pool_{kernel_size}", mp.get_shape())
            pooled.append(mp)

        state = tf.concat(pooled, axis=3)
        state = tf.squeeze(state, axis=[1, 2])
        print("per_story_states", state.get_shape())
        return state

    def _optimizer(self) -> tf.train.Optimizer:
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def _output_fc(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
        x = tf.layers.dense(x, 300, activation=tf.nn.leaky_relu)
        x = tf.layers.dropout(x, rate=self.keep_prob, training=self.is_training)
        output = tf.layers.dense(x, self.CLASSES, activation=None, name="output")
        print("output", output.get_shape())
        return output
