import inspect
import os

import numpy as np
import tensorflow as tf

from . import model as model_module
from .data.dataset import Dataset
from .flags import define_flags


def train(network, dset, batch_size=1, epochs=1):
    network.train(data=dset, batch_size=batch_size, epochs=epochs)


def test(network, dset, batch_size, expname):
    predictions = network.predict_epoch(dset, "test", batch_size)
    pred_dir = os.path.join(network.save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    predictions_fname = os.path.join(pred_dir, f"{os.path.basename(FLAGS.checkpoint_path)}_exp{expname}_test.txt")
    with open(predictions_fname, "w+") as f:
        for p in predictions:
            print(p, file=f)


def main(FLAGS):
    print("Loading data...")
    dset = Dataset(FLAGS.train_file, FLAGS.eval_file, FLAGS.test_file)
    print("Data shapes:", dset.train_data.shape, dset.eval_data.shape, dset.test_data.shape)
    print("Vocabulary size:", len(dset.vocab))

    print("Initializing network...")
    network = None

    for name, obj in inspect.getmembers(model_module):
        if inspect.isclass(obj) and name == FLAGS.model:
            network = obj(*FLAGS)

    if network is None:
        raise ValueError(f"Unknown model {FLAGS.model}.")

    if FLAGS.checkpoint_path is None:
        print("Running network...")
        train(network, dset, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
    else:
        print("Testing...")
        test(network, dset, batch_size=FLAGS.batch_size, expname=FLAGS.exp)


if __name__ == "__main__":
    define_flags()
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    np.random.seed(FLAGS.seed)
    main(FLAGS)
