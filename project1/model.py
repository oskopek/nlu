from collections import Counter
import datetime
import os
import sys

from gensim import models
import numpy as np
import tensorflow as tf


######################### CONSTANTS #####################################


ROOT = "."
DATA_FOLDER = "data"
CONTINUATION = os.path.join(DATA_FOLDER, "sentences.continuation")
EVAL = os.path.join(DATA_FOLDER, "sentences.eval")
TRAIN = os.path.join(DATA_FOLDER, "sentences.train")
EMBEDDINGS = os.path.join(DATA_FOLDER, "pretrained_embeddings")

LOG_DIR = os.path.join(ROOT, "logs")
SAVE_DIR = os.path.join(ROOT, "checkpoints")

SENTENCE_LEN = 30
PREDICT_LEN = 20
VOCABULARY_LEN = 20_000
EMBEDDING_DIM = 100
LSTM_DIM = 512
INTERMEDIATE_DIM = 512
BATCH_SIZE = 64
GRAD_CLIP = 5
SEED = 42
BOS_SYMBOL = "<bos>"
EOS_SYMBOL = "<eos>"
PAD_SYMBOL = "<pad>"
UNK_SYMBOL = "<unk>"
BASE_VOCAB = {UNK_SYMBOL: 0, BOS_SYMBOL: 1, EOS_SYMBOL: 2, PAD_SYMBOL: 3}


######################### EMBEDDINGS #####################################


# Copied from:
# http://da.inf.ethz.ch/teaching/2018/NLU/material/load_embeddings.py
def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.items():
        tok_augmented = tok[1:-1]
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        elif tok_augmented in model.vocab:
            print("Using embedding of '{}' for '{}'".format(tok_augmented, tok))
            external_embedding[idx] = model[tok_augmented]
            matches += 1
        else:
            print("%s not in embedding file, initializing randomly" % tok)
            external_embedding[idx] = np.random.uniform(
                low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    # here, embeddings are actually set
    session.run(assign_op, {pretrained_embeddings: external_embedding})


######################### DATASET #####################################


# Replace missing values with the default value, but do not insert them.
class missingdict(dict):

    def __init__(self, default_val=None, *args, **kwargs):
        super(missingdict, self).__init__(*args, **kwargs)
        self.default_val = default_val

    def __missing__(self, key):
        return self.default_val


class Dataset:
    train_data = None
    eval_data = None
    continuation_data = None
    embedding_file = None
    vocab = None
    inv_vocab = None

    @staticmethod
    def read_lines(file):
        with open(file, "r") as f:
            lines = f.readlines()
        return lines

    @staticmethod
    def generate_vocab(lines):
        counter = Counter()
        for line in lines:
            split = line.strip().split(" ")
            counter.update(split)

        vocab = dict(BASE_VOCAB)
        id = 4
        for word, _ in counter.most_common(VOCABULARY_LEN - 4):
            vocab[word] = id
            id += 1

        vocab = missingdict(vocab[UNK_SYMBOL], vocab)
        return vocab

    @staticmethod
    def encode_words(lines, vocab, padding_size=SENTENCE_LEN):
        data = np.zeros((len(lines), padding_size), dtype=np.int32)

        used_counter = 0
        for line in lines:
            split = line.strip().split(" ")
            if len(split) <= padding_size - 2:
                parsed_line = np.full((padding_size), vocab[PAD_SYMBOL], dtype=np.int32)

                split.insert(0, BOS_SYMBOL)
                split.append(EOS_SYMBOL)
                idxes = np.asarray(list(map(lambda word: vocab[word], split))).astype(np.int32)

                parsed_line[:len(idxes)] = idxes
                data[used_counter, :] = parsed_line
                used_counter += 1

        # trim last lines that are only zeros (of sentences longer than 30)
        data = data[:used_counter, :]
        return data

    def __init__(self, train_file, eval_file, continuation_file, embedding_file):
        # Reading training has to happen first!
        self.train_data = self.read_data(train_file)
        self.eval_data = self.read_data(eval_file)

        # Only read continuations into lines, because we do not pad them.
        self.continuation_lines = Dataset.read_lines(continuation_file)
        self.embedding_file = embedding_file

    def read_data(self, file):
        lines = Dataset.read_lines(file)
        if self.vocab is None:
            self.vocab = Dataset.generate_vocab(lines)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            print("Vocabulary size:", len(self.vocab))
            assert VOCABULARY_LEN == len(self.vocab)
        data = Dataset.encode_words(lines, self.vocab)
        return data

    def load_embeddings(self, session, emb_matrix, dim_embedding):
        assert VOCABULARY_LEN == len(self.vocab)
        load_embedding(session,
                       self.vocab,
                       emb_matrix,
                       self.embedding_file,
                       dim_embedding,
                       len(self.vocab))

    def batches_per_epoch_generator(self, batch_size, data=None, shuffle=True):
        if data is None:
            data = self.train_data

        n_rows = data.shape[0]
        if shuffle:
            train_permutation = np.random.permutation(n_rows)
        else:
            train_permutation = np.arange(n_rows)

        for i in range(0, n_rows, batch_size):
            yield data[train_permutation[i: i + batch_size]]


######################### NETWORK #####################################


class NetworkUtils:

    @staticmethod
    def trainable_zero_state(batch_size, lstm_dim, reuse=False):
        with tf.variable_scope("rnn_zero_state", reuse=reuse):
            state1 = tf.get_variable(
                name="intial_state_c",
                shape=[lstm_dim],
                dtype=tf.float32)
            state2 = tf.get_variable(
                name="intial_state_m",
                shape=[lstm_dim],
                dtype=tf.float32)
            state1 = tf.reshape(tf.tile(state1, [batch_size]), (-1, lstm_dim))
            state2 = tf.reshape(tf.tile(state2, [batch_size]), (-1, lstm_dim))
            print("state_real", state1.get_shape())
            return (state1, state2)

    @staticmethod
    def tee(x):
        def print_fnc(x):
            print(x)
            return np.zeros(shape=1, dtype=np.float32)
        return x + tf.py_func(print_fnc, [x], tf.float32)

    @staticmethod
    def dense_layer(x, dims, name=None):
        # TODO: Perhaps write our own function for this?
        return tf.layers.dense(x, dims, use_bias=False, name=name)  # TODO: Perhaps use bias?

    @staticmethod
    def calc_perplexity(probs, indices):
        def dynamic_mean(values, lens, axis=1):
            return tf.reduce_sum(values, axis=axis) / lens
        with tf.name_scope("perplexity"):
            print("probs", probs.get_shape())
            print("indices", indices.get_shape())
            epsilon = 1e-8
            mask = tf.cast(tf.not_equal(indices, BASE_VOCAB[PAD_SYMBOL]), tf.float32)
            print("mask shape", mask.get_shape())
            sentence_lens = tf.reduce_sum(mask, axis=1)
            return tf.exp(-dynamic_mean(tf.log(tf.maximum(probs, epsilon)) * mask, lens=sentence_lens, axis=1))


class Network:
    session = None
    summary_writer = None
    lstm_dim = None
    words_input = None
    embedding_matrix = None
    word_embeddings = None
    loss = None
    trainer = None
    train_summaries = None
    test_summaries = None
    initializer = tf.contrib.layers.xavier_initializer
    dataset = None
    save_path = None
    save_avg_perplexity = None
    save_avg_loss = None
    global_step = None
    zero_state = None
    log_dir = None
    rnn_cell = None

    def output_layer(self, x, reuse=False):
        with tf.variable_scope("output_layer", reuse=reuse):
            if self.lstm_dim != INTERMEDIATE_DIM:
                x = NetworkUtils.dense_layer(x, INTERMEDIATE_DIM, name="hidden_layer")
            return NetworkUtils.dense_layer(x, VOCABULARY_LEN, name="softmax_layer")

    def create_sentences(self, name, indices):
        def lookup(indices):
            # print("pyfunc_input", indices.shape)
            result = []
            for n_batch in range(indices.shape[0]):
                # print("indices[n_batch]", indices[n_batch].shape)
                batch_result = [self.dataset.inv_vocab[i] for i in indices[n_batch]]
                batch_unpadded = []
                for word in batch_result:
                    if word.strip() == EOS_SYMBOL.strip():
                        break
                    batch_unpadded.append(word)
                result.append(" ".join(batch_unpadded))
            result = ["{}. \x60{}\x60".format(i, r) for i, r in enumerate(result)]
            return "\n".join(result)
        sentences = tf.py_func(lookup, [indices], tf.string)
        return tf.summary.text(name, sentences)

    def __init__(self, dataset, log_dir=None, save_path=None, embedding_dim=EMBEDDING_DIM,
                 lstm_dim=LSTM_DIM, load_embeddings=False, train_init_state=True, restore_from=None):
        graph = tf.Graph()
        graph.seed = SEED
        self.dataset = dataset
        self.lstm_dim = lstm_dim
        self.save_path = save_path
        self.session = tf.Session(graph=graph)

        with self.session.graph.as_default():
            self.words_input = tf.placeholder(tf.int32, (None, SENTENCE_LEN), name="words_input")

            self._embeddings(load_embeddings=load_embeddings, embedding_dim=embedding_dim)  # Embeddings
            self._rnn(train_init_state=train_init_state)  # Unrolled RNN
            self._man()  # Manual step-by-step generation
            self._summaries()  # Summaries
            self._optimizer()  # Add optimizer nodes
            self._savers(log_dir=log_dir)  # Summary writer and checkpoint saver

            if restore_from is not None:  # restore
                self.saver.restore(self.session, restore_from)
            else:  # init
                self.session.run(tf.global_variables_initializer())

    def _embeddings(self, load_embeddings, embedding_dim):
        with tf.name_scope("embeddings"):
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=[
                    VOCABULARY_LEN, embedding_dim], dtype=tf.float32)
            if load_embeddings:
                self.dataset.load_embeddings(self.session, self.embedding_matrix, embedding_dim)
            self.word_embeddings = tf.nn.embedding_lookup(
                self.embedding_matrix, self.words_input, name="embedding_lookup")
            print("word_embeddings", self.word_embeddings.get_shape())

    def _rnn(self, train_init_state):
        with tf.name_scope("rnn"):
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(
                num_units=self.lstm_dim,
                initializer=self.initializer(),
                state_is_tuple=True)

            # Zeros state for RNN.
            batch_size = tf.shape(self.words_input)[0]
            state_shape = tf.stack([batch_size, self.lstm_dim], axis=0)
            if train_init_state:
                state = NetworkUtils.trainable_zero_state(batch_size, self.lstm_dim)
            else:
                state = (tf.zeros(state_shape, name="zero_state_c"),
                         tf.zeros(state_shape, name="zero_state_m"))
            self.zero_state1, self.zero_state2 = state
            self.loss = tf.zeros(batch_size)

            # RNN for loop
            next_word_probs = []
            pred_indices = []
            range_batch_size = tf.range(batch_size)
            with tf.name_scope("rnnloop"):
                for i in range(SENTENCE_LEN - 1):
                    word, next_word_index = self.word_embeddings[:, i, :], self.words_input[:, i + 1]
                    x, state = self.rnn_cell(word, state)
                    x = self.output_layer(x, reuse=(i > 0))
                    probs = tf.nn.softmax(x, name="softmax_probs")
                    pred_indices.append(tf.argmax(x, axis=1))
                    indices_to_gather = tf.stack([range_batch_size, next_word_index], axis=1)
                    # print("indices_to_gather", indices_to_gather.get_shape())
                    p = tf.gather_nd(probs, indices_to_gather)
                    # print("word_probs_p", p.get_shape())
                    next_word_probs.append(p)
                    mask = tf.cast(tf.not_equal(next_word_index, BASE_VOCAB[PAD_SYMBOL]), tf.float32)
                    self.loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=next_word_index, name="word_loss") * mask

            self.pred_indices = tf.stack(pred_indices, axis=1)
            print("pred_indices", self.pred_indices.get_shape())
            next_word_probs = tf.stack(next_word_probs, axis=1)
            print("stacked_probs", next_word_probs.get_shape())
            indices = self.words_input[:, 1:]
            self.perplexity = NetworkUtils.calc_perplexity(next_word_probs, indices)

            mask = tf.cast(tf.not_equal(indices, BASE_VOCAB[PAD_SYMBOL]), tf.float32)
            print("mask shape", mask.get_shape())
            sentence_lens = tf.reduce_sum(mask, axis=1)
            self.loss /= sentence_lens
            self.mean_loss = tf.reduce_mean(self.loss)
            print("loss", self.loss.get_shape(), self.loss.dtype)
            print("mean_loss", self.mean_loss.get_shape(), self.mean_loss.dtype)

    def _man(self):
        with tf.name_scope("man"):
            self.man_word_index = tf.placeholder(tf.int32, (), name="word_index")
            word_emb = tf.nn.embedding_lookup(self.embedding_matrix,
                                              self.man_word_index,
                                              name="embedding_lookup")
            word_emb = tf.expand_dims(word_emb, axis=0)
            print("word_emb", word_emb.get_shape())

            self.man_state1 = tf.placeholder(tf.float32, (1, self.lstm_dim), name="init_state_c")
            self.man_state2 = tf.placeholder(tf.float32, (1, self.lstm_dim), name="init_state_m")
            state = (self.man_state1, self.man_state2)
            x, state = self.rnn_cell(word_emb, state)
            x = self.output_layer(x, reuse=True)
            self.man_out_word_index = tf.argmax(x, axis=1)
            self.man_out_state1, self.man_out_state2 = state

    def _summaries(self):
        train_perplexity_summ = tf.summary.scalar("train/avg_perplexity", tf.reduce_mean(self.perplexity))
        train_perplexity2_summ = tf.summary.scalar("train/avg_perplexity_xentropy", tf.exp(self.mean_loss))
        train_loss_summ = tf.summary.scalar("train/loss", self.mean_loss)
        # The text prediction summaries don't work with lower versions of TF:
        train_text_truth_summ = self.create_sentences("train/ground_truth", self.words_input[:, 1:])
        train_text_predict_summ = self.create_sentences("train/predicted", self.pred_indices)

        train_summaries = [train_perplexity_summ, train_perplexity2_summ, train_loss_summ, train_text_truth_summ, train_text_predict_summ]
        self.train_summaries = tf.summary.merge(train_summaries, name="train_summaries")

        self.save_test_perplexity = tf.placeholder(tf.float32, [], name="save_test_perplexity")
        self.save_test_loss = tf.placeholder(tf.float32, [], name="save_test_loss")
        test_avg_perplexity_summ = tf.summary.scalar("test/avg_perplexity", self.save_test_perplexity)
        test_avg_loss_summ = tf.summary.scalar("test/loss", self.save_test_loss)
        test_summaries = [test_avg_perplexity_summ, test_avg_loss_summ]
        self.test_summaries = tf.summary.merge(test_summaries, name="test_summaries")

    def _optimizer(self):
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer()
            with tf.name_scope("clip"):
                gradients = optimizer.compute_gradients(self.mean_loss)
                clipped_gradients = [(tf.clip_by_norm(gradient, GRAD_CLIP), var) for gradient, var in gradients]
            self.trainer = optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)

    def _savers(self, log_dir):
        with tf.name_scope("savers"):
            self.summary_writer = tf.summary.FileWriter(log_dir)
            self.summary_writer.add_graph(self.session.graph)
            self.saver = tf.train.Saver()

    def run_batch(self, inputs, train, monitor):
        targets = [self.global_step, self.mean_loss]
        if monitor:
            targets.append(self.perplexity)
            if train:
                targets.append(self.train_summaries)
        if train:
            targets.append(self.trainer)

        outputs = self.session.run(targets, feed_dict={self.words_input: inputs})

        if monitor and train:
            self.summary_writer.add_summary(outputs[3], global_step=outputs[0])

        if monitor:
            return outputs[1:3]
        return outputs[1]

    def eval(self, dataset, batch_size):
        batch_losses, sentence_perplexities = [], []
        for batch in dataset.batches_per_epoch_generator(batch_size, data=dataset.eval_data, shuffle=False):
            batch_loss, batch_perplexities = self.run_batch(batch, train=False, monitor=True)
            batch_losses.append(batch_loss)
            sentence_perplexities.extend(batch_perplexities)
        loss, perplexity = np.mean(batch_losses), np.mean(sentence_perplexities)
        # Save the results in Tensorboard
        step, summs = self.session.run([self.global_step, self.test_summaries],
                                       feed_dict={self.save_test_perplexity: perplexity,
                                                  self.save_test_loss: loss})
        self.summary_writer.add_summary(summs, global_step=step)
        print("Batches", step, "finished. Loss:", loss, "Perplexity:", perplexity)
        sys.stdout.flush()  # Flush output, so that bpeek works.
        return np.array(sentence_perplexities)

    def finish_sentence(self, sentence, zero_states, predict_len):
        def create_feed_dict(word, state1, state2):
            return {self.man_word_index: word,
                    self.man_state1: state1,
                    self.man_state2: state2}

        last_word = None
        fetches = [self.man_out_word_index, self.man_out_state1, self.man_out_state2]
        result = []

        # Get initial state. Use dummy input to infer batch size.
        state1, state2 = zero_states
        assert len(sentence) <= 20
        for word in sentence:
            last_word, state1, state2 = self.session.run(fetches, feed_dict=create_feed_dict(word, state1, state2))
            last_word = last_word[0]
            result.append(word)

        assert last_word is not None
        while len(result) < predict_len and last_word != BASE_VOCAB[EOS_SYMBOL]:
            last_word, state1, state2 = self.session.run(fetches, feed_dict=create_feed_dict(last_word, state1, state2))
            last_word = last_word[0]
            result.append(last_word)
        return result

    def finish_sentences(self, dataset, sentences, predict_len):
        init_state1, init_state2 = self.session.run([self.zero_state1, self.zero_state2], feed_dict={
            self.words_input: np.zeros((1, SENTENCE_LEN))})
        result = []
        for i, sentence in enumerate(sentences):
            # Encode the sentence and trim the EOS symbol
            sentence_len = len(sentence.strip().split(" "))
            word_indices = Dataset.encode_words([sentence], dataset.vocab, padding_size=(sentence_len + 2))[0, :-1]

            init_state1_c = np.array(init_state1, copy=True, dtype=np.float32)
            init_state2_c = np.array(init_state2, copy=True, dtype=np.float32)
            out_indices = self.finish_sentence(word_indices, [init_state1_c, init_state2_c], predict_len)
            finished_sentence = [dataset.inv_vocab[word] for word in out_indices]
            result.append(finished_sentence)
            if i % 500 == 0:
                print(i, "out of", len(sentences), "done:", finished_sentence)
        return result

    def run(self, dataset, batch_size, epochs):
        for epoch in range(epochs):
            print("Epoch", epoch)
            for n_batch, batch in enumerate(dataset.batches_per_epoch_generator(batch_size, data=dataset.train_data)):
                self.run_batch(batch, train=True, monitor=n_batch % 50 == 0)

            perplexities = self.eval(dataset, batch_size)
            # Save the model and all the perplexities
            self.saver.save(self.session, self.save_path, global_step=epoch)
            with open("{}-{}-perplex.txt".format(self.save_path, epoch), "w+") as f:
                for p in perplexities:
                    print(p, file=f)


######################### EXPERIMENTS #####################################


def gen_expname(expname):
    return "{}-{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"), expname)


def expA():
    expname = gen_expname("LSTM512-RNN")
    network = Network(dset, lstm_dim=512, embedding_dim=100, load_embeddings=False,
                      log_dir=os.path.join(LOG_DIR, expname),
                      save_path=os.path.join(SAVE_DIR, expname))
    return network


def expB():
    expname = gen_expname("LSTM512-RNN-w2v-emb")
    return Network(dset, lstm_dim=512, embedding_dim=100, load_embeddings=True,
                   log_dir=os.path.join(LOG_DIR, expname),
                   save_path=os.path.join(SAVE_DIR, expname))


def expC():
    expname = gen_expname("LSTM1024-RNN-w2v-emb")
    return Network(dset, lstm_dim=1024, embedding_dim=100, load_embeddings=True,
                   log_dir=os.path.join(LOG_DIR, expname),
                   save_path=os.path.join(SAVE_DIR, expname))


def expD(checkpoint_path):
    expname = gen_expname("LSTM1024-RNN-w2v-emb-generate")
    network = Network(dset, lstm_dim=1024, embedding_dim=100, load_embeddings=False,
                      log_dir=os.path.join(LOG_DIR, expname),
                      save_path=os.path.join(SAVE_DIR, expname),
                      restore_from=checkpoint_path)
    sentences = network.finish_sentences(dataset=dset, sentences=dset.continuation_lines, predict_len=PREDICT_LEN)
    with open(os.path.join(SAVE_DIR, "{}_gen.txt".format(os.path.basename(checkpoint_path))), "w+") as f:
        for s in sentences:
            print(" ".join(s), file=f)


def main(args):
    print("Loading data...")
    global dset
    dset = Dataset(TRAIN, EVAL, CONTINUATION, EMBEDDINGS)
    print("Data shapes:", dset.train_data.shape, dset.eval_data.shape, len(dset.continuation_lines))
    print("Vocabulary size:", len(dset.vocab))

    print("Initializing network...")
    experiment = args.exp.strip().lower()

    network = None
    if experiment == "a":
        network = expA()
    elif experiment == "b":
        network = expB()
    elif experiment == "c":
        network = expC()
    elif experiment == "d":
        expD(args.checkpoint_path)
        return
    else:
        raise ValueError("Unknown Experiment \'{}\'.".format(experiment))

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    print("Running network...")
    network.run(dset, BATCH_SIZE, EPOCHS)


if __name__ == "__main__":
    np.random.seed(SEED)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--exp", default="a", help="Experiment to run.")
    parser.add_argument("--checkpoint_path", default="", help="Full checkpoint path.")
    args = parser.parse_args()
    main(args)
