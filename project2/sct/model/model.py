from collections import OrderedDict
from datetime import datetime
import os

import tensorflow as tf
import tqdm


class Model:
    SENTENCES = 4
    ENDINGS = 2

    def _placeholders(self):
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

        self.sentence_lens = tf.placeholder(tf.int32, [None, self.SENTENCES], name="sentence_lens")
        self.ending_lens = tf.placeholder(tf.int32, [None, self.ENDINGS], name="ending_lens")

        self.sentence_word_ids = tf.placeholder(tf.int32, [None, self.SENTENCES, None], name="sentence_word_ids")
        self.ending_word_ids = tf.placeholder(tf.int32, [None, self.ENDINGS, None], name="ending_word_ids")

        self.sentence_charseq_ids = tf.placeholder(tf.int32, [None, self.SENTENCES, None], name="sentence_charseq_ids")
        self.ending_charseq_ids = tf.placeholder(tf.int32, [None, self.ENDINGS, None], name="ending_charseq_ids")

        self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
        self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        self.is_training = tf.placeholder_with_default(False, [], name="is_training")

    def _summaries_and_init(self):
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

    def __init__(self, *args, threads=1, seed=42, logdir="logs", expname="exp", **kwargs):
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

    def build_model(self):
        """
        Remember to use `with self.session.graph.as_default():`.

        :return: `predictions`, `loss`, and `training_step`.
        """
        raise NotImplementedError("To be overridden.")

    def _build_feed_dict(self, batch, is_training=False):
        (sentence_lens, ending_lens), (sentence_word_ids, ending_word_ids), (sentence_charseq_ids, ending_charseq_ids),\
            charseqs, charseq_lens, labels = batch
        return {
                self.sentence_lens: sentence_lens,
                self.ending_lens: ending_lens,
                self.sentence_word_ids: sentence_word_ids,
                self.ending_word_ids: ending_word_ids,
                self.sentence_charseq_ids: sentence_charseq_ids,
                self.ending_charseq_ids: ending_charseq_ids,
                self.charseqs: charseqs,
                self.charseq_lens: charseq_lens,
                self.labels: labels,
                self.is_training: is_training
        }

    @staticmethod
    def _tqdm_metrics(dataset, metrics, names):
        d = OrderedDict()
        assert len(metrics) == len(names)
        for metric, name in zip(metrics, names):
            d[f'{dataset}_{name}'] = metric
        return d

    def train_batch(self, batch):
        self.session.run(self.reset_metrics)
        fetches = [self.current_metrics, self.training_step, self.summaries["train"]]
        metrics, *_ = self.session.run(fetches, self._build_feed_dict(batch, is_training=True))
        return self._tqdm_metrics("train", metrics, ["acc", "loss"])

    def train(self, data, epochs, batch_size=1):

        def _eval_metrics():
            dataset = "eval"
            eval_metrics = self.evaluate_epoch(data.eval_data, dataset, batch_size=batch_size)
            return self._tqdm_metrics(dataset, eval_metrics, ["acc", "loss"])

        epoch_tqdm = tqdm.tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_tqdm:
            batch_count, batch_generator = data.batches_per_epoch(batch_size, data=data.train_data)
            batch_tqdm = tqdm.tqdm(range(batch_count), desc=f"Batches [Epoch {epoch}]")
            for _ in batch_tqdm:
                batch = next(batch_generator)
                metrics = self.train_batch(batch)
                batch_tqdm.set_postfix(metrics)
            epoch_tqdm.set_postfix(_eval_metrics())

    def evaluate_epoch(self, data, dataset, batch_size=1):
        self.session.run(self.reset_metrics)
        for batch in data.batches_per_epoch_generator(batch_size, data=data):
            self.session.run(self.update_metrics, self._build_feed_dict(batch))
        metrics, _ = self.session.run(self.current_metrics + [self.summaries[dataset]])
        return metrics

    def predict_epoch(self, data, dataset, batch_size=1):
        self.session.run(self.reset_metrics)
        predictions = []
        self.session.run(self.reset_metrics)
        for batch in data.batches_per_epoch_generator(batch_size, data=data):
            batch_predictions = self.session.run(self.predictions, self._build_feed_dict(batch))
            predictions.extend(batch_predictions)
        self.session.run(self.summaries[dataset])
        return predictions
