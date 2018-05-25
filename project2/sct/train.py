import inspect
import os
import sys

import numpy as np
import tensorflow as tf

from . import flags
from . import model as model_module
from .data.datasets import Datasets
from .data.preprocessing import Preprocessing


def train(network: model_module.Model, dsets: Datasets, batch_size: int = 1, epochs: int = 1) -> None:
    network.train(data=dsets, batch_size=batch_size, epochs=epochs)


def test(network: model_module.Model, dsets: Datasets, batch_size: int = 1, expname: str = "exp") -> None:
    predictions = network.predict_epoch(dsets.test, "test", batch_size)
    pred_dir = os.path.join(network.save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    predictions_fname = os.path.join(pred_dir, f"{os.path.basename(FLAGS.checkpoint_path)}_exp{expname}_test.txt")
    with open(predictions_fname, "w+") as f:
        for p in predictions:
            print(p, file=f)

    # TODO(oskopek): Proper output formatting.


def main(FLAGS: tf.app.flags._FlagValuesWrapper) -> None:
    print("Loading data...", flush=True)
    preprocessing = Preprocessing(standardize=True,)
    dsets = Datasets(FLAGS.train_file, FLAGS.eval_file, FLAGS.test_file, preprocessing=preprocessing)

    print("Initializing network...", flush=True)
    network = None

    for name, obj in inspect.getmembers(model_module):
        if inspect.isclass(obj) and name == FLAGS.model:
            vocabularies = dsets.train.vocabularies
            num_sentences = len(vocabularies.sentence_vocabulary)
            num_words = len(vocabularies.word_vocabulary)
            num_chars = len(vocabularies.char_vocabulary)
            flag_dict = {k: v.value for k, v in {**FLAGS.__flags}.items()}  # HACK
            network = obj(num_sentences=num_sentences, num_words=num_words, num_chars=num_chars, **flag_dict)

    if network is None:
        raise ValueError(f"Unknown model {FLAGS.model}.")

    if FLAGS.checkpoint_path is None:
        print("Running network...", flush=True)
        train(network, dsets, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
    else:
        print("Testing...", flush=True)
        test(network, dsets, batch_size=FLAGS.batch_size, expname=FLAGS.exp)
    print("End.")
    print("EndStdErr.", file=sys.stderr)


if __name__ == "__main__":
    flags.define_flags()
    FLAGS = tf.app.flags.FLAGS

    np.random.seed(FLAGS.seed)
    main(FLAGS)
