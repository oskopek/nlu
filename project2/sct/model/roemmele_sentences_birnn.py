import tensorflow as tf

from .roemmele_sentences import RoemmeleSentences


class RoemmeleSentencesBiRNN(RoemmeleSentences):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _sentence_rnn(self, per_sentence_states: tf.Tensor) -> tf.Tensor:
        assert len(per_sentence_states.get_shape()) == 3
        assert per_sentence_states.get_shape()[1] == self.TOTAL_SENTENCES - 1
        cell_fw = self._create_cell(self.rnn_cell_dim, name='sentence_cell_fw')
        # cell_bw = self._create_cell(self.rnn_cell_dim, name='sentence_cell_bw')
        cell_bw = cell_fw

        inputs = per_sentence_states
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs, dtype=tf.float32)
        if self.rnn_cell == "LSTM":
            state_fw = state_fw[0]  # c_state
            state_bw = state_bw[0]  # c_state
        per_story_states = tf.concat([state_bw, state_fw], axis=1)
        print("per_story_states", per_story_states.get_shape())
        return per_story_states

    def _output_fc(self, x: tf.Tensor) -> tf.Tensor:
        output = tf.layers.dense(x, self.CLASSES, activation=None, name="output")
        print("output", output.get_shape())
        return output
