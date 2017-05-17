"""

Tensorflow implementation of WV-DR algorithm wrapper class(sklearn style)

:authro: Fido Wang (wanghailin317@gmail.com)
:refer: http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

"""
import tensorflow as tf
import numpy as np
from option import Option
import math
import time
import collections
import random


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
        data_index = (data_index + 1) % len(word_ids)
        if i == batch_size:
            yield batch, labels[:, None]
            i = 0
            labels = np.ndarray(shape=(batch_size), dtype=np.int32)
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)


class Skipgram(object):
    """
    Skip-Gram word embedder

    Support Functional Style setting thus all function begin with 'set' and 'build'
    will return an object,the Embedding Estimator itself.
    """

    def __init__(self, options):
        assert isinstance(options, Option)
        self._options = options
        self._session = None
        self.saver = None

    def setVocab(self, vocab):
        self.vocab = vocab
        return self

    def setTrainData(self, train, labels):
        self.train = train
        self.labels = labels
        return self

    def useSubSampling(self, switch=True, threshold=1e-5):
        self.use_sub_sampling = switch
        self.sub_sampling_threshold = 1e-5
        return self

    def _get_inputs(self):
        """
        Create TF Placeholders for input, targets, and learning rate.
        :return: Tuple of Placeholders (input, targets, learning rate)
        """
        inputs_ = tf.placeholder(tf.int32, [None], name='input')
        labels_ = tf.placeholder(tf.int32, [None, None], name='target')
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
        return tf.nn.embedding_lookup(embedding, input_data)

    def build_graph(self):
        """
        Create Graph and Initialize tf Session for training
        """
        train_graph = tf.Graph()
        opts = self._options
        with train_graph.as_default():
            self.__inputs, self.__labels, self.__lr = self._get_inputs()
            embeddings = self._get_embedding_layer(self.__inputs)

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
                                                  inputs=embeddings,
                                                  num_sampled=opts.negative_sample_size,
                                                  num_classes=opts.vocab_size)
                tf.summary.scalar("Softmax loss", loss)
            elif opts.loss == 'nce':
                loss = tf.nn.nce_loss(weights=weights,
                                      biases=biases,
                                      labels=self.__labels,
                                      inputs=embeddings,
                                      num_sampled=opts.negative_sample_size,
                                      num_classes=opts.vocab_size)
                tf.summary.scalar("NCE loss", loss)
            self.__cost = tf.reduce_mean(loss)

            if opts.train_method == 'Adam':
                self.__optimizer = tf.train.AdamOptimizer(self.__lr).minimize(self.__cost)
            else:
                self.__optimizer = tf.train.GradientDescentOptimizer(self.__lr).minimize(self.__cost)

            self._session = tf.Session(graph=train_graph)
            self.saver = tf.train.Saver()
        return self

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

    def _get_target(self, words, idx, window_size=5):
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
                for x, y in batches:
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
                        print("Epoch {}/{}".format(e, opts.epochs_to_train + 10),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(period_loss),
                              "{:.4f} sec/batch".format((end - start) * 1.0 / opts.statistics_interval))
                        loss = 0
                        start = time.time()
                    if iteration % opts.checkpoint_interval == 0:
                        self.saver.save(self._session,
                                        "word2vec",
                                        global_step=iteration)
                    iteration += 1
            self._word_embeddings = self.__normalized_word_embeddings.eval()
            self.saver.save(self._session, "final_word2vec")

    def transform_w(self, word_index):
        return self._word_embeddings[word_index, :]

    def transform_doc(self, word_indexs):
        doc_embeddings = [self._word_embeddings[i, :] for i in word_indexs]
        return doc_embeddings
