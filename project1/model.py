import os
import tensorflow as tf
import numpy as np
from collections import Counter
from gensim import models

DATA_FOLDER = "./data"
CONTINUATION = os.path.join(DATA_FOLDER, "sentences.continuation")
EVAL = os.path.join(DATA_FOLDER, "sentences.eval")
TRAIN = os.path.join(DATA_FOLDER, "sentences.train")
EMBEDDINGS = os.path.join(DATA_FOLDER, "pretrained_embeddings")
LOG_DIR = os.path.join('.', 'logs')

SENTENCE_LEN = 30
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


class missingdict(dict):
    
    def __init__(self, default_val=None, *args, **kwargs):
      super(missingdict, self).__init__(*args, **kwargs)
      self.default_val = default_val
  
    def __missing__(self, key):
        return self.default_val


# Copied from: http://da.inf.ethz.ch/teaching/2018/NLU/material/load_embeddings.py

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
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, vocab_size))
    
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding}) # here, embeddings are actually set


class Dataset:
  train_data = None
  eval_data = None
  continuation_data = None
  embedding_file = None
  vocab = None
  inv_vocab = None
  
  def read_lines(file):
    with open(file, "r") as f:
      lines = f.readlines()
    return lines

  def generate_vocab(lines):
    counter = Counter()
    for line in lines:
      split = line.strip().split(" ")
      counter.update(split)

    vocab = dict(BASE_VOCAB)
    id = 4
    for word, occurences in counter.most_common(VOCABULARY_LEN - 4):
      vocab[word] = id
      id += 1

    vocab = missingdict(vocab[UNK_SYMBOL], vocab)
    return vocab

  def encode_words(lines, vocab, padding_size=SENTENCE_LEN):
    data = np.zeros((len(lines), padding_size), dtype=np.int32)

    used_counter = 0
    for line in lines:
      split = line.strip().split(" ")
      if len(split) < padding_size - 2:
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
  
  def read_data(self, file):
    lines = Dataset.read_lines(file)
    if self.vocab is None:
      self.vocab = Dataset.generate_vocab(lines)
      self.inv_vocab = {v: k for k, v in self.vocab.items()}
      assert VOCABULARY_LEN == len(self.vocab)
    data = Dataset.encode_words(lines, self.vocab)
    return data
 
  def __init__(self, train_file, eval_file, continuation_file, embedding_file):
    self.train_data = self.read_data(train_file) # Reading training has to happen first!
    self.eval_data = self.read_data(eval_file)
    self.continuation_data = self.read_data(continuation_file)
    self.embedding_file = embedding_file
    
  def load_embeddings(self, session, emb_matrix, dim_embedding):
    assert VOCABULARY_LEN == len(self.vocab)
    load_embedding(session, self.vocab, emb_matrix, self.embedding_file, dim_embedding, len(self.vocab))
    
  def batches_per_epoch_generator(self, batch_size, data=None):
    if data is None:
      data = self.train_data
    
    n_rows = data.shape[0]
    train_permutation = np.random.permutation(n_rows)
    
    for i in range(0, n_rows, batch_size):
      yield data[train_permutation[i : i + batch_size]]
    


dset = Dataset(TRAIN, EVAL, CONTINUATION, EMBEDDINGS)
print(dset.train_data.shape)
print(dset.eval_data.shape)
print(dset.continuation_data.shape)
print(len(dset.vocab))
print(dset.eval_data)
for i, batch in enumerate(dset.batches_per_epoch_generator(1000, data=dset.eval_data)):
    print(i, batch.shape)




class Network:
  session = None
  summary_writer = None
  lstm_dim = None
  words_input = None
  loss = None
  trainer = None
  train_summaries = None
  test_summaries = None
  initializer = tf.contrib.layers.xavier_initializer
  dataset = None
  
  def tee(self, x):
    def print_fnc(x):
      print(x)
      return np.zeros(shape=1, dtype=np.float32)
    return x+tf.py_func(print_fnc, [x], tf.float32)
  
  def trainable_zero_state(self, batch_size, lstm_dim):
    state1 = tf.get_variable(name="rnn_intial_state_c", shape=[lstm_dim], dtype=tf.float32)
    state2 = tf.get_variable(name="rnn_intial_state_m", shape=[lstm_dim], dtype=tf.float32)
    state1 = tf.reshape(tf.tile(state1, [batch_size]), (-1, lstm_dim))
    state2 = tf.reshape(tf.tile(state2, [batch_size]), (-1, lstm_dim))
    print("state_real", state1.get_shape())
    return (state1, state2)
  
  def dense_layer(self, x, dims, name=None):
    # TODO: Perhaps write our own function for this? 
    with self.session.graph.as_default():
      return tf.layers.dense(x, dims, use_bias=False, name=name)
  
  def output_layer(self, x, reuse=False): # Use tf.AUTO_REUSE when newer tensorflow
    with self.session.graph.as_default():
      with tf.variable_scope("output_layer", reuse=reuse):
        if self.lstm_dim != INTERMEDIATE_DIM:
          x = self.dense_layer(x, INTERMEDIATE_DIM, name="hidden_layer")
        return self.dense_layer(x, VOCABULARY_LEN, name="softmax_layer")

  def calc_perplexity(self, probs, indices):
    print("probs", probs.get_shape())
    print("indices", indices.get_shape())
    epsilon = epsilon = 1e-8
    mask = indices != BASE_VOCAB[PAD_SYMBOL]
    #probs = self.tee(probs)
    return tf.exp(-tf.reduce_mean(tf.log(probs + epsilon) * mask, axis=1))
  
  def create_sentences(self, name, indices):
    def lookup(indices):
      # print("pyfunc_input", indices.shape)
      result = []
      for n_batch in range(indices.shape[0]):
        # print("indices[n_batch]", indices[n_batch].shape)
        batch_result = [self.dataset.inv_vocab[i] for i in indices[n_batch]]
        result.append(' '.join(batch_result))
      return '\n'.join(result)
    sentences = tf.py_func(lookup, [indices], tf.string)
    return tf.summary.text(name, sentences)
    
  
  def __init__(self, dataset, log_dir=None, embedding_dim=EMBEDDING_DIM, vocab_len=VOCABULARY_LEN, lstm_dim=LSTM_DIM, train_embeddings=True, train_init_state=True):
    graph = tf.Graph()
    graph.seed = SEED
    self.dataset = dataset
    self.session = tf.Session(graph=graph)

    with self.session.graph.as_default():
      self.lstm_dim = lstm_dim
      self.words_input = tf.placeholder(tf.int32, (None, SENTENCE_LEN), name="words_input")

      self.summary_writer = tf.summary.FileWriter(log_dir)

      # Embeddings
      embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[VOCABULARY_LEN, embedding_dim], dtype=tf.float32, trainable=train_embeddings)
      if not train_embeddings:
        self.dataset.load_embeddings(self.session, embedding_matrix, embedding_dim)
      word_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.words_input, name="embedding_lookup")
      print("word_embeddings", word_embeddings.get_shape())

      rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_dim, initializer=self.initializer(), state_is_tuple=True)

      # Zeros state for RNN.
      batch_size = tf.shape(self.words_input)[0]
      state_shape = tf.stack([batch_size, lstm_dim], axis=0)
      if train_init_state:
        state = self.trainable_zero_state(batch_size, lstm_dim)
      else:
        state = (tf.zeros(state_shape, name="rnn_intial_state_c"),
                 tf.zeros(state_shape, name="rnn_intial_state_m"))
      self.loss = 0

      # RNN for loop
      next_word_probs = []
      pred_indices = []
      range_batch_size = tf.range(batch_size)
      for i in range(SENTENCE_LEN-1):
        word, next_word_index = word_embeddings[:,i,:], self.words_input[:,i+1]
        x, state = rnn_cell(word, state)
        x = self.output_layer(x, reuse=i>0)
        probs = tf.nn.softmax(x, name="softmax_probs")
        pred_indices.append(tf.argmax(x, axis=1))
        indices_to_gather = tf.stack([range_batch_size, next_word_index], axis=1)
        #print("indices_to_gather", indices_to_gather.get_shape())
        p = tf.gather_nd(probs, indices_to_gather)
        #print("word_probs_p", p.get_shape())
        next_word_probs.append(p)
        self.loss += tf.losses.compute_weighted_loss(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=next_word_index, name="partial_loss"))

      next_word_probs = tf.stack(next_word_probs, axis=1)
      print("stacked_probs", next_word_probs.get_shape())
      pred_indices = tf.stack(pred_indices, axis=1)
      print("pred_indices", pred_indices.get_shape())
      self.perplexity = self.calc_perplexity(next_word_probs, self.words_input[:,1:])
      print("loss", self.loss.get_shape(), self.loss.dtype)
      
      # Summaries
      train_perplexity_summ = tf.summary.scalar("train/avg_perplexity", tf.reduce_mean(self.perplexity))
      train_loss_summ = tf.summary.scalar("train/loss", self.loss)
      train_text_truth_summ = self.create_sentences("train/ground_truth", self.words_input[:, 1:])
      train_text_predict_summ = self.create_sentences("train/predicted", pred_indices)
      self.train_summaries = tf.summary.merge([train_perplexity_summ,
                                               train_loss_summ
                                               # train_text_truth_summ, 
                                               # train_text_predict_summ
                                               ], name="train_summaries")
      test_perplexity_summ = tf.summary.scalar("test/avg_perplexity", tf.reduce_mean(self.perplexity))
      test_loss_summ = tf.summary.scalar("test/loss", self.loss)
      test_text_truth_summ = self.create_sentences("test/ground_truth", self.words_input[:, 1:])
      test_text_predict_summ = self.create_sentences("test/predicted", pred_indices)
      self.test_summaries = tf.summary.merge([test_perplexity_summ,
                                              test_loss_summ
                                              # test_text_truth_summ, 
                                              # test_text_predict_summ
                                              ], name="test_summaries")

      # Optimizer
      optimizer = tf.train.AdamOptimizer()
      self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
      gradients = optimizer.compute_gradients(self.loss)
      clipped_gradients = [(tf.clip_by_norm(gradient, 5), var) for gradient, var in gradients]
      self.trainer = optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)

      self.summary_writer.add_graph(self.session.graph)
      self.session.run(tf.global_variables_initializer())
    
  def run_batch(self, inputs, train, monitor):
    targets = [self.global_step, self.loss]
    if monitor:
      targets.append(self.perplexity)
      targets.append(self.train_summaries if train else self.test_summaries)
    if train:
      targets.append(self.trainer)
    
    outputs = self.session.run(targets, feed_dict={self.words_input: inputs})
    
    if monitor:
      self.summary_writer.add_summary(outputs[3], global_step=outputs[1])
    
    if monitor:
      return outputs[1:3]
    return outputs[1]
  
  def run(self, dataset, batch_size, epochs):
    for epoch in range(epochs):
      print("Epoch", epoch)
      for n_batch, batch in enumerate(dataset.batches_per_epoch_generator(batch_size)):
        self.run_batch(batch, train=True, monitor=n_batch % 50 == 0)
      ls = []
      ps = []
      for n_batch, batch in enumerate(dataset.batches_per_epoch_generator(batch_size, data=dset.eval_data)):
        l, p = self.run_batch(batch, train=False, monitor=True)
        ls.append(l)
        ps.extend(p)
      print("Epoch", epoch, "finished. Loss:", np.mean(ls), "Perplexity:", np.mean(ps))
        




import datetime
expname = "{}-LSTM-RNN".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
network = Network(dset, lstm_dim=512, embedding_dim=100, log_dir=os.path.join(LOG_DIR, expname))
EPOCHS = 10
network.run(dset, BATCH_SIZE, EPOCHS)
