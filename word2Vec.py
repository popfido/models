"""

Tensorflow implementation of WV-DR algorithm wrapper class(sklearn style)

:authro: Fido Wang (wanghailin317@gmail.com)
:refer: http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

"""
import tensorflow as tf
import os
import numpy as np
from option import Option
from functools import reduce
import math
import time
import collections
import random
import pickle
from error import NotTrainedError


def generate_batch_para(doc_ids, word_ids, batch_size, num_skips, window_size):
    """
    batch generator for Skip-Gram Model(Distributed Representation of Word Vecotors)
    :param doc_ids: list of document indices
    :param word_ids: list of word indices
    :param batch_size: number of words in each mini-batch
    :param num_skips: number of sample for each target word window
    :param window_size: number of words between the target word
    """
    data_index = 0
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * window_size
    labels = np.ndarray(shape=(batch_size), dtype=np.int32)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = 2 * window_size + 1
    buffer = collections.deque(maxlen=span)
    buffer_para = collections.deque(maxlen=span)

    i = 0
    while data_index < len(word_ids):
        if len(buffer) == span and len(set(buffer_para)) == 1:
            target = window_size
            targets_to_avoid = [window_size]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                labels[i + j] = buffer[target]
                batch[i + j] = buffer[window_size]
            i += num_skips
        buffer.append(word_ids[data_index])
        buffer_para.append(doc_ids[data_index])
        # data_index = (data_index + 1) % len(word_ids)
        data_index = (data_index + 1)
        if i == batch_size:
            yield batch, labels[:, None], doc_ids[data_index]
            i = 0
            labels = np.ndarray(shape=(batch_size), dtype=np.int32)
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)


class Skipgram(object):
    """
    Skip-Gram word embedder

    Support Functional Style setting thus all function begin with 'set' and 'build'
    will return an object,the Embedding Estimator itself.
    """

    def __init__(self, options, vocab=None):
        assert isinstance(options, Option)
        self._options = options
        self.__inputs, self.__labels, self.__lr = None, None, None
        self.word_embeddings = None
        self.__normalized_word_embeddings = None
        self.__cost, self.__optimizer, self. __summary = None, None ,None
        self.vocab = vocab
        self.reversed_vocab = None if vocab is None else {v: k for k,v in vocab.items()}
        self.use_subsampling = False
        self._session = None
        self.saver = None

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.reversed_vocab = None if vocab is None else {v: k for k, v in vocab.items()}
        return self

    def use_subsampling(self, switch=True):
        self.use_subsampling = switch
        return self

    def _get_inputs(self):
        """
        Create TF Placeholders for input, targets, and learning rate.
        :return: Tuple of Placeholders (input, targets, learning rate)
        """
        opts = self._options
        inputs_ = tf.placeholder(tf.int32, [opts.batch_size], name='input')
        labels_ = tf.placeholder(tf.int32, [opts.batch_size, 1], name='target')
        lr_ = tf.placeholder(tf.float32, name='learning_rate')
        return inputs_, labels_, lr_

    def _get_embedding_layer(self, input_data):
        """
        Create embedding for <input_data>.
        :param input_data: TF placeholder for text input.
        :return: Embedded input.
        """
        opts = self._options
        embedding = tf.Variable(tf.random_uniform((opts.vocab_size, opts.embed_dim), -1, 1))
        return embedding, tf.nn.embedding_lookup(embedding, input_data)

    def build_graph(self):
        """
        Create Graph and Initialize tf Session for training
        """
        train_graph = tf.Graph()
        self.graph = train_graph
        opts = self._options
        with train_graph.as_default():
            self.__inputs, self.__labels, self.__lr = self._get_inputs()
            embeddings, embed = self._get_embedding_layer(self.__inputs)

            norm_w = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.__normalized_word_embeddings = embeddings / norm_w

            weights = tf.Variable(
                tf.truncated_normal((opts.vocab_size, opts.embed_dim),
                                    stddev=1.0 / math.sqrt(opts.embed_dim))
            )
            biases = tf.Variable(tf.zeros(opts.vocab_size))
            if opts.loss == 'softmax':
                loss = tf.nn.sampled_softmax_loss(weights=weights,
                                                  biases=biases,
                                                  labels=self.__labels,
                                                  inputs=embed,
                                                  num_sampled=opts.negative_sample_size,
                                                  num_classes=opts.vocab_size)
            elif opts.loss == 'nce':
                loss = tf.nn.nce_loss(weights=weights,
                                      biases=biases,
                                      labels=self.__labels,
                                      inputs=embed,
                                      num_sampled=opts.negative_sample_size,
                                      num_classes=opts.vocab_size)
            self.__cost = tf.reduce_mean(loss)
            tf.summary.scalar("w2v_loss", self.__cost)

            if opts.train_method == 'Adam':
                self.__optimizer = tf.train.AdamOptimizer(self.__lr).minimize(self.__cost)
            else:
                self.__optimizer = tf.train.GradientDescentOptimizer(self.__lr).minimize(self.__cost)
            self.__summary = tf.summary.merge_all()
            self._session = tf.Session(graph=train_graph)
            self.saver = tf.train.Saver()

        return self

    @staticmethod
    def _get_batches(self, doc_ids, word_ids, batch_size, skip_size, window_size):
        n_batches = len(word_ids) // batch_size
        words = word_ids[:n_batches * batch_size]
        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx + batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = self._get_target(batch, ii, window_size)
                y.extend(batch_y)
                x.extend([batch_x] * len(batch_y))
            yield np.array(x), np.array(y)[:, None]

    @staticmethod
    def _get_target(words, idx, window_size=5):
        R = np.random.randint(1, window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = set(words[start:idx] + words[idx + 1:stop + 1])

        return list(target_words)

    def _choose_batch_generator(self, chosen='words'):
        if chosen == 'words':
            self._generate_batches = self._get_batches
        if chosen == 'docs':
            self._generate_batches = generate_batch_para

    def fit(self, train_data):
        opts = self._options
        iteration = 1
        loss = 0
        if type(train_data[0]) is int:
            word_ids = train_data
            doc_ids = [0] * len(train_data)
            self._choose_batch_generator(chosen='words')
        elif type(train_data[0]) is list:
            doc_ids = [[i] * len(j) for i, j in enumerate(train_data)]
            doc_ids = [item for sublist in doc_ids for item in sublist]
            word_ids = [item for sublist in train_data for item in sublist]
            self._choose_batch_generator(chosen='docs')

        with self._session as session:
            session.run(tf.global_variables_initializer())
            for e in range(1, opts.epochs_to_train):
                batches = self._generate_batches(doc_ids, word_ids, opts.batch_size, opts.window_size, opts.window_size)
                start = time.time()
                learning_rate = opts.learning_rate
                for x, y, train_idx in batches:
                    period_loss = 1000000000
                    learning_rate = opts.learning_rate if learning_rate < period_loss else learning_rate * 0.1
                    feed = {self.__inputs: x,
                            self.__labels: y,
                            self.__lr: learning_rate}
                    train_loss, _ = session.run([self.__cost, self.__optimizer], feed_dict=feed)

                    loss += train_loss
                    if iteration % opts.statistics_interval == 0:
                        period_loss = loss / opts.statistics_interval
                        end = time.time()
                        print("Epoch {}/{}".format(e, opts.epochs_to_train - 1),
                              "Iteration: {}".format(iteration),
                              "Current Doc: {}".format(train_idx),
                              "Avg. Training loss: {:.4f}".format(period_loss),
                              "{:.4f} sec/batch".format((end - start) * 1.0 / opts.statistics_interval))
                        loss = 0
                        start = time.time()
                    if iteration % opts.checkpoint_interval == 0:
                        self.saver.save(self._session,
                                        "word2vec",
                                        global_step=iteration)
                    iteration += 1
            self.word_embeddings = self.__normalized_word_embeddings.eval()
            self.saver.save(self._session, "final_word2vec")

    def transform_w(self, word_index):
        if self.word_embeddings is None:
            raise NotTrainedError
        return self.word_embeddings[word_index, :]

    def transform_doc(self, word_indexes):
        if self.word_embeddings is None:
            raise NotTrainedError
        opts = self._options
        sample = opts.subsample
        doc_len = 0
        doc = []
        for idx in word_indexes:
            #ran = (math.sqrt(self.vocab[idx] / sample * len(self.vocab)) + 1) * (sample * len(self.vocab)) / self.vocab[idx]
            #if ran > random.random():
            doc_len += 1
            doc.append(self.word_embeddings[idx, :])

        doc_embeddings = reduce(lambda x,y: x+y, doc)/doc_len
        return doc_embeddings

    def save(self, path):
        """
        To save trained model and its params.
        """
        save_path = self.saver.save(self._session,'model.data')
        # save parameters of the model
        params = self._options
        pickle.dump(params,
                  open(os.path.join(path, 'model_params.json'), 'wb'), pickle.HIGHEST_PROTOCOL)

        # save dictionary, reverse_dictionary
        pickle.dump(self.vocab,
                  open(os.path.jooin(path, 'model_dict.json'), 'wb'), pickle.HIGHEST_PROTOCOL)

        print("Model saved in file: %s" % save_path)
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self._session, path)
        with self._session as session:
            session.run(tf.global_variables_initializer())
            self.word_embeddings = self.__normalized_word_embeddings.eval()

    @classmethod
    def restore(cls, path):
        """
        To restore a saved model.
        """
        # load params of the model
        path_dir = os.path.dirname(path)
        params = pickle.load(open(os.path.join(path_dir, 'model_params.json'), 'rb'))
        # init an instance of this class
        estimator = Skipgram(params)
        estimator.build_graph();
        estimator._restore(os.path.join(path_dir, 'model.data'))
        # bind dictionaries
        estimator.set_vocab(pickle.load(open(os.path.join(path_dir, 'model_dict.json'), 'rb')))

        return estimator
