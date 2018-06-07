import os

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def define_flags() -> None:
    # Directories
    flags.DEFINE_string('logdir', 'outputs', 'Logdir name.')
    flags.DEFINE_string('expname', 'Roemmele_Sentences_VAN_Rand6', 'Experiment name.')

    flags.DEFINE_string('checkpoint_path', None, 'Checkpoint to load. If none, ignored.')

    # Data files
    flags.DEFINE_string('train_file', 'data/stories.train.csv', 'Train data file.')
    flags.DEFINE_string('eval_file', 'data/stories.eval.csv', 'Evaluation data file.')
    flags.DEFINE_list('test_files', ['data/stories.test.csv', 'data/stories.spring2016.csv'], 'Test data files.')
    flags.DEFINE_string('skip_thought_folder', os.environ['SCRATCH'] + '/st', 'Skip-thought embeddings folder.')

    # Model choice
    flags.DEFINE_string(
            'model', 'RoemmeleSentences',
            'Model class name. Models that have "sentences" in their name have different data preprocessing steps.')
    flags.DEFINE_integer('roemmele_multiplicative_factor', 6, 'How many negative endings to sample. Need 1 for '
                                                              '`add` not None.')
    flags.DEFINE_string('add', None, 'Whether and which constant add to use for negative labels.')
    flags.DEFINE_bool('eval_train', False, 'Train on first 80% of eval dataset, eval on rest.')
    flags.DEFINE_bool('balanced_batches', False, 'Train with label-balanced batches.')
    flags.DEFINE_string('attention', None, 'Attention type (add ~ Bahdanau, mult ~ Luong, None). Only for Roemmele '
                        'models.')
    flags.DEFINE_integer('attention_size', 1000, 'Attention size.')

    # TF parameters
    flags.DEFINE_boolean("no_gpu", False, 'Disables GPU usage even if a GPU is available')
    flags.DEFINE_integer('threads', 8, 'Maximum number of threads to use.')
    flags.DEFINE_integer('seed', 42, 'Random seed')

    # Optimization parameters
    flags.DEFINE_integer('epochs', 10, 'Training epoch count')
    flags.DEFINE_integer('batch_size', 100, 'Training batch size')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('grad_clip', 10.0, 'Gradient clipped to L2 norm smaller than or equal to.')

    # Jupyter notebook params
    # Only to avoid raising UnrecognizedFlagError
    flags.DEFINE_string('f', 'kernel', 'Kernel')

    # Other
    flags.DEFINE_string('rnn_cell', "VAN", 'RNN cell type. If None, attention-only model.')
    flags.DEFINE_integer('rnn_cell_dim', 1000, 'RNN cell dimension.')
    flags.DEFINE_integer('word_embedding', 620, 'word_embedding')
    flags.DEFINE_integer('char_embedding', 200, 'char_embedding')
    flags.DEFINE_integer('sentence_embedding', 4800, 'sentence_embedding')
    flags.DEFINE_float('keep_prob', 0.5, 'dropout probability')
