from typing import List

import inspect
import os
import sys

import numpy as np
import tensorflow as tf

from . import flags
from . import model as model_module
from .data.datasets import Datasets
from .data.preprocessing import Preprocessing
from .data.stories import NLPStoriesDataset


def train(network: model_module.Model, dsets: Datasets, batch_size: int = 1, epochs: int = 1) -> None:
    network.train(data=dsets, batch_size=batch_size, epochs=epochs, evaluate_every_steps=FLAGS.evaluate_every_steps)


def test(network: model_module.Model, dsets: Datasets, batch_size: int = 1, expname: str = "expname") -> None:
    if network.last_checkpoint_path is not None:  # do not load on external checkpoint load
        print("Restoring last/best model...", flush=True)
        network.restore_last()
    pred_dir = os.path.join(network.save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    for stories, test_fname in zip(dsets.tests, dsets.test_files):
        test_fname = os.path.basename(test_fname)
        predictions, accuracy = network.predict_epoch(stories, "test", batch_size)
        predictions = [1 + p for p in predictions]  # map back to labels
        fname = os.path.join(pred_dir, f"pred_exp-{expname}_test-{test_fname}.txt")
        if accuracy is not None:
            print(f"{fname} accuracy:\t{accuracy}")
        print_output(predictions, fname)


def print_output(predictions: List[int], fname: str) -> None:
    with open(fname, "w+") as f:
        for p in predictions:
            print(p, file=f)


def main(FLAGS: tf.app.flags._FlagValuesWrapper) -> None:
    print("Loading data...", flush=True)
    preprocessing = Preprocessing(standardize=True,)
    dsets = Datasets(
            FLAGS.train_file,
            FLAGS.eval_file,
            FLAGS.test_files,
            skip_thought_folder=FLAGS.skip_thought_folder,
            preprocessing=preprocessing,
            roemmele_multiplicative_factor=FLAGS.roemmele_multiplicative_factor,
            eval_train=FLAGS.eval_train,
            balanced_batches=FLAGS.balanced_batches,
            sent_embedding='sentences' in FLAGS.model.lower(),  # HACK
    )

    print("Initializing network...", flush=True)
    network = None

    for name, obj in inspect.getmembers(model_module):
        if inspect.isclass(obj) and name == FLAGS.model:
            flag_dict = {k: v.value for k, v in {**FLAGS.__flags}.items()}  # HACK
            if isinstance(dsets.train, NLPStoriesDataset):
                vocabularies = dsets.train.vocabularies
                flag_dict['num_sentences'] = len(vocabularies.sentence_vocabulary)
                flag_dict['num_words'] = len(vocabularies.word_vocabulary)
                flag_dict['num_chars'] = len(vocabularies.char_vocabulary)
            network = obj(**flag_dict)

    if network is None:
        raise ValueError(f"Unknown model {FLAGS.model}.")

    print("Load checkpoint:", FLAGS.load_checkpoint)
    if FLAGS.load_checkpoint is not None:  # loading checkpoint
        print("Loading checkpoint...")
        network.restore(FLAGS.load_checkpoint)
    else:  # training
        print("Running network...", flush=True)
        train(network, dsets, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)

    print("Testing...", flush=True)
    test(network, dsets, batch_size=FLAGS.batch_size, expname=FLAGS.expname)

    print("End.")
    print("EndStdErr.", file=sys.stderr)


if __name__ == "__main__":
    flags.define_flags()
    FLAGS = tf.app.flags.FLAGS

    np.random.seed(FLAGS.seed)
    main(FLAGS)
