from collections import OrderedDict
from datetime import datetime
import os
from typing import Tuple, Dict, Union, List, Any

from sct.data.datasets import Datasets
from sct.data.stories import StoriesDataset

from dotmap import DotMap
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
        self.batch_to_sentences = tf.placeholder(tf.int32, [None, self.TOTAL_SENTENCES], name="batch_to_sentences")

        # [unique sentence_ids, max_word_ids]
        self.sentence_to_words = tf.placeholder(tf.int32, [None, None], name="sentence_to_words")
        self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")

        # [unique word_ids, max_char_ids]
        self.word_to_chars = tf.placeholder(tf.int32, [None, None], name="word_to_chars")
        self.word_lens = tf.placeholder(tf.int32, [None], name="word_lens")

        # [batch_size]
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        self.is_training = tf.placeholder_with_default(False, [], name="is_training")

    def _summaries_and_init(self) -> None:
        current_accuracy, update_accuracy = tf.metrics.accuracy(self.labels, self.predictions)
        current_loss, update_loss = tf.metrics.mean(self.loss)
        self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
        self.current_metrics = [current_accuracy, current_loss]
        self.update_metrics = [update_accuracy, update_loss]

        summary_writer = tf.contrib.summary.create_file_writer(self.save_dir, flush_millis=5_000)
        self.summaries = {}
        with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
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

        # Initialize variables
        self.session.run(tf.initialize_all_variables())
        with summary_writer.as_default():
            tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def __init__(self, *args, threads: int = 1, seed: int = 42, logdir: str = "logs", expname: str = "exp", **kwargs):
        self.save_dir = os.path.join(f"{logdir}", f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}-{expname}")

        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        config = {
                "gpu_options": tf.GPUOptions(allow_growth=True),
                "inter_op_parallelism_threads": threads,
                "intra_op_parallelism_threads": threads
        }
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(*config))

        # Construct the graph
        with self.session.graph.as_default():
            self._placeholders()
            self.predictions, self.loss, self.training_step = self.build_model()
            self._summaries_and_init()

    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Operation]:
        """
        Remember to use `with self.session.graph.as_default():`.

        :return: `predictions`, `loss`, and `training_step`.
        """
        raise NotImplementedError("To be overridden.")

    def _build_feed_dict(self, batch: DotMap[str, Union[np.ndarray, bool]],
                         is_training: bool = False) -> Dict[tf.Tensor, Union[np.ndarray, bool]]:
        assert is_training == batch.is_training
        return {
                self.batch_to_sentences: batch.batch_to_sentences,
                self.sentence_to_words: batch.sentence_to_words,
                self.sentence_lens: batch.sentence_lens,
                self.word_to_chars: batch.word_to_chars,
                self.word_lens: batch.word_lens,
                self.labels: batch.labels,
                self.is_training: batch.is_training
        }

    @staticmethod
    def _tqdm_metrics(dataset: str, metrics: List[Any], names: List[str]) -> Dict[str, str]:
        d = OrderedDict()
        assert len(metrics) == len(names)
        for metric, name in zip(metrics, names):
            d[f'{dataset}_{name}'] = str(metric)
        return d

    def train_batch(self, batch) -> Dict[str, str]:
        self.session.run(self.reset_metrics)
        fetches = [self.current_metrics, self.training_step, self.summaries["train"]]
        metrics, *_ = self.session.run(fetches, self._build_feed_dict(batch, is_training=True))
        return self._tqdm_metrics("train", metrics, ["acc", "loss"])

    def train(self, data: Datasets, epochs: int, batch_size: int = 1) -> None:

        def _eval_metrics() -> Dict[str, str]:
            dataset = "eval"
            eval_metrics = self.evaluate_epoch(data.eval, dataset, batch_size=batch_size)
            return self._tqdm_metrics(dataset, eval_metrics, ["acc", "loss"])

        epoch_tqdm = tqdm.tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_tqdm:
            batch_count, batch_generator = data.train.batches_per_epoch(batch_size)
            batch_tqdm = tqdm.tqdm(range(batch_count), desc=f"Batches [Epoch {epoch}]")
            for _ in batch_tqdm:
                batch = next(batch_generator)
                metrics = self.train_batch(batch)
                batch_tqdm.set_postfix(metrics)
            epoch_tqdm.set_postfix(_eval_metrics())

    def evaluate_epoch(self, data: StoriesDataset, dataset: str, batch_size: int = 1) -> List[float]:
        self.session.run(self.reset_metrics)
        for batch in data.batch_per_epoch_generator(batch_size, shuffle=False):
            self.session.run(self.update_metrics, self._build_feed_dict(batch))
        metrics, _ = self.session.run(self.current_metrics + [self.summaries[dataset]])
        return metrics

    def predict_epoch(self, data: StoriesDataset, dataset: str, batch_size: int = 1) -> List[int]:
        self.session.run(self.reset_metrics)
        predictions = []
        self.session.run(self.reset_metrics)
        for batch in data.batch_per_epoch_generator(batch_size, shuffle=False):
            batch_predictions = self.session.run(self.predictions, self._build_feed_dict(batch))
            predictions.extend(batch_predictions)
        self.session.run(self.summaries[dataset])
        return predictions
