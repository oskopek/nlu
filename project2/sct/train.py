import inspect
import os
from typing import *

import numpy as np
import tensorflow as tf

from . import model as model_module
from .data.datasets import Datasets
from .flags import define_flags


def train(network: model_module.Model, dsets: Datasets, batch_size: int = 1, epochs: int = 1) -> None:
    network.train(data=dsets, batch_size=batch_size, epochs=epochs)


def test(network: model_module.Model, dsets: Datasets, batch_size: int = 1, expname: str = "exp") -> None:
    predictions = network.predict_epoch(dsets.test_file, "test", batch_size)
    pred_dir = os.path.join(network.save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    predictions_fname = os.path.join(pred_dir, f"{os.path.basename(FLAGS.checkpoint_path)}_exp{expname}_test.txt")
    with open(predictions_fname, "w+") as f:
        for p in predictions:
            print(p, file=f)


def main(FLAGS: tf.app.flags._FlagValuesWrapper[str, Any]) -> None:
    print("Loading data...")
    dsets = Datasets(FLAGS.train_file, FLAGS.eval_file, FLAGS.test_file)

    print("Initializing network...")
    network = None

    for name, obj in inspect.getmembers(model_module):
        if inspect.isclass(obj) and name == FLAGS.model:
            network = obj(*FLAGS)

    if network is None:
        raise ValueError(f"Unknown model {FLAGS.model}.")

    if FLAGS.checkpoint_path is None:
        print("Running network...")
        train(network, dsets, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
    else:
        print("Testing...")
        test(network, dsets, batch_size=FLAGS.batch_size, expname=FLAGS.exp)


if __name__ == "__main__":
    define_flags()
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    np.random.seed(FLAGS.seed)
    main(FLAGS)
