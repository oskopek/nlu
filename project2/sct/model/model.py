from collections import OrderedDict
from datetime import datetime
import os
from typing import Tuple, Dict, Union, List, Any

from sct.data.datasets import Datasets
from sct.data.stories import StoriesDataset

import numpy as np
import tensorflow as tf
import tqdm


class Model:
    SENTENCES = 4
    ENDINGS = 2
    TOTAL_SENTENCES = SENTENCES + ENDINGS

    def _placeholders(self) -> None:
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

        # [batch_size, SENTENCES x sentence_id]
        self.batch_to_sentence_ids = tf.placeholder(
                tf.int32, [None, self.TOTAL_SENTENCES], name="batch_to_sentence_ids")
        self.batch_to_sentences = tf.placeholder(tf.int32, [None, self.TOTAL_SENTENCES], name="batch_to_sentences")

        # [unique sentence ids, max_word_ids]
        self.sentence_to_word_ids = tf.placeholder(tf.int32, [None, None], name="sentence_to_word_ids")
        self.sentence_to_words = tf.placeholder(tf.int32, [None, None], name="sentence_to_words")
        # [unique sentence ids]
        self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")

        # [unique word ids, max_char_ids]
        self.word_to_char_ids = tf.placeholder(tf.int32, [None, None], name="word_to_chars")
        # [unique word ids]
        self.word_lens = tf.placeholder(tf.int32, [None], name="word_lens")

        # [batch_size]
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        # [] bool scalar
        self.is_training = tf.placeholder_with_default(False, [], name="is_training")

        # Useful tensors
        self.batch_size = tf.shape(self.batch_to_sentence_ids)[0]

    def _summaries_and_init(self) -> None:
        with tf.name_scope("summaries"):
            current_accuracy, update_accuracy = tf.metrics.accuracy(self.labels, self.predictions)
            current_loss, update_loss = tf.metrics.mean(self.loss)
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
            self.current_metrics = [current_accuracy, current_loss]
            self.update_metrics = [update_accuracy, update_loss]

            summary_writer = tf.contrib.summary.create_file_writer(self.save_dir, flush_millis=10_000)
            self.summaries: Dict[str, List[tf.Operation]] = dict()
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(50):
                self.summaries["train"] = [
                        tf.contrib.summary.scalar("train/loss", update_loss),
                        tf.contrib.summary.scalar("train/accuracy", update_accuracy)
                ]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["eval", "test"]:
                    self.summaries[dataset] = [
                            tf.contrib.summary.scalar(dataset + "/loss", current_loss),
                            tf.contrib.summary.scalar(dataset + "/accuracy", current_accuracy)
                    ]

        # Saver
        self.saver = tf.train.Saver(max_to_keep=20)

        # Initialize variables
        self.session.run(tf.global_variables_initializer())
        with summary_writer.as_default():
            tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def __init__(self,
                 num_sentences: int,
                 num_words: int,
                 num_chars: int,
                 *args,
                 threads: int = 1,
                 seed: int = 42,
                 logdir: str = "logs",
                 expname: str = "exp",
                 **kwargs) -> None:
        super().__init__()
        self.num_sentences = num_sentences
        self.num_words = num_words
        self.num_chars = num_chars
        self.save_dir = os.path.join(f"{logdir}", f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-{expname}")
        self.checkpoint_path = os.path.join(self.save_dir, "checkpoints", "model.ckpt")

        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        config = {
                "gpu_options": tf.GPUOptions(allow_growth=True),
                "inter_op_parallelism_threads": threads,
                "intra_op_parallelism_threads": threads
        }
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(**config))

        # Construct the graph
        with self.session.graph.as_default():
            self._placeholders()
            self.predictions, self.loss, self.training_step = self.build_model()
            self._summaries_and_init()

    def save(self) -> str:
        return self.saver.save(self.session, self.checkpoint_path)

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        """
        Remember to use `with self.session.graph.as_default():`.

        :return: `predictions`, `loss`, and `training_step`.
        """
        raise NotImplementedError("To be overridden.")

    @staticmethod
    def _tqdm_metrics(dataset: str, metrics: List[Any], names: List[str]) -> Dict[str, str]:
        d: Dict[str, str] = OrderedDict()
        assert len(metrics) == len(names)
        for metric, name in zip(metrics, names):
            d[f'{dataset}_{name}'] = str(metric)
        return d

    def _train_metrics(self) -> Dict[str, str]:
        train_metrics = self.session.run(self.current_metrics)
        return Model._tqdm_metrics("train", train_metrics, ["acc", "loss"])

    def _eval_metrics(self, data: Datasets, batch_size: int = 1) -> Dict[str, str]:
        dataset = "eval"
        eval_metrics = self.evaluate_epoch(data.eval, dataset, batch_size=batch_size)
        return Model._tqdm_metrics(dataset, eval_metrics, ["acc", "loss"])

    def _build_feed_dict(self, batch: Dict[str, Union[np.ndarray, bool]],
                         is_training: bool = False) -> Dict[tf.Tensor, Union[np.ndarray, bool]]:
        assert is_training == batch['is_training']
        return {
                self.batch_to_sentence_ids: batch['batch_to_sentence_ids'],
                self.batch_to_sentences: batch['batch_to_sentences'],
                self.sentence_to_word_ids: batch['sentence_to_word_ids'],
                self.sentence_to_words: batch['sentence_to_words'],
                self.sentence_lens: batch['sentence_lens'],
                self.word_to_char_ids: batch['word_to_char_ids'],
                # self.word_to_chars: batch['word_to_chars'],
                self.word_lens: batch['word_lens'],
                self.labels: batch['labels'],
                self.is_training: batch['is_training']
        }

    def train_batch(self, batch: Dict[str, Union[np.ndarray, bool]]) -> List[Any]:
        self.session.run(self.reset_metrics)
        fetches = [self.training_step, self.summaries["train"]]
        return self.session.run(fetches, self._build_feed_dict(batch, is_training=True))

    def _train(self, data: Datasets, epochs: int, batch_size: int = 1) -> None:
        with tqdm.tqdm(range(epochs), desc="Epochs") as epoch_tqdm:
            for epoch in epoch_tqdm:
                batch_count, batch_generator = data.train.batches_per_epoch(batch_size)
                with tqdm.tqdm(range(batch_count), desc=f"Batches [Epoch {epoch}]") as batch_tqdm:
                    for _ in batch_tqdm:
                        batch = next(batch_generator)
                        self.train_batch(batch)
                        # Can be enabled, but takes up time during training
                        # batch_tqdm.set_postfix(self._train_metrics())
                epoch_tqdm.set_postfix(self._eval_metrics(data, batch_size=batch_size))
                self.save()

    def train(self, data: Datasets, epochs: int, batch_size: int = 1) -> None:
        try:
            self._train(data, epochs, batch_size=batch_size)
        except KeyboardInterrupt as e:
            print("Cancelling training and saving model...")
            print(f"Done: {self.save()}")

    def evaluate_epoch(self, data: StoriesDataset, dataset: str, batch_size: int = 1) -> List[float]:
        self.session.run(self.reset_metrics)
        for batch in data.batch_per_epoch_generator(batch_size, shuffle=False):
            self.session.run(self.update_metrics, self._build_feed_dict(batch))
        returns = self.session.run(self.current_metrics + [self.summaries[dataset]])
        return returns[:len(self.current_metrics)]  # return current metrics

    def predict_epoch(self, data: StoriesDataset, dataset: str, batch_size: int = 1) -> List[int]:
        self.session.run(self.reset_metrics)
        predictions: List[int] = []
        self.session.run(self.reset_metrics)
        for batch in data.batch_per_epoch_generator(batch_size, shuffle=False):
            batch_predictions = self.session.run(self.predictions, self._build_feed_dict(batch))
            predictions.extend(batch_predictions)
        self.session.run(self.summaries[dataset])
        return predictions
