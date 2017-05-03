"""

Tensorflow implementation of Doc2VecC algorithm wrapper class

:author: Fido Wang (wanghailin317@gmail.com)
:refer: https://openreview.net/pdf?id=B1Igu2ogg

"""

import tensorflow as tf
import numpy as np
from option import Option
import math
import time
import collections
from itertools import compress
import random


def generate_batch_doc2VecC_tail(doc_ids, word_ids, doc_len, batch_size, window_size, max_sample, sentence_sample):
    """
    batch generator for PV-DM (Distribbuted Memory Model of Paragraph Vectors)
    :param doc_ids: list of document indices
    :param word_ids: list of word indices
    :param batch_size: number of words in each mini-batch
    :param window_size: number of words before the target word
    :return: tuple of (batch, labels, batch_doc_sample, num_sampled)
    """
    data_index = 0
    assert batch_size % window_size == 0
    batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    batch_doc = np.ndarray(shape=(batch_size, max_sample), dtype=np.int32)
    num_sampled = np.ndarray(shape=(batch_size), dtype=np.int32)
    span = window_size + 1
    buffer = collections.deque(maxlen=span)
    buffer_doc = collections.deque(maxlen=span)
    mask = [1] * span
    mask[-1] = 0
    i = 0

    while data_index < len(word_ids):
        if len(set(buffer_doc)) == 1 and len(buffer_doc) == span:
            doc_id = buffer_doc[-1]
            batch[i, :] = list(compress(buffer, mask)) + [doc_id]
            labels[i, 0] = buffer[-1]

            # Sample the sentence
            already_sampled = 0
            for ele in word_ids[doc_len[doc_id]:doc_len[doc_id + 1]]:
                if already_sampled == max_sample:
                    break
                if random.random() < sentence_sample:
                    batch_doc[i, already_sampled] = ele
                    already_sampled += 1
            if already_sampled == 0:
                batch_doc[i, 0] = word_ids[random.randint(doc_len[doc_id], doc_len[doc_id + 1] - 1)]
                already_sampled = 1

            num_sampled[i] = already_sampled
            i += 1
        buffer.append(word_ids[data_index])
        buffer_doc.append(doc_ids[data_index])
        data_index = (data_index + 1) % len(word_ids)
        if i == batch_size:
            yield batch, labels, batch_doc, num_sampled
            batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
            labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
            batch_doc = np.ndarray(shape=(batch_size, max_sample), dtype=np.int32)
            num_sampled = np.ndarray(shape=(batch_size), dtype=np.int32)
            i = 0


class Doc2VecC(object):
    """
    Doc2VecC embedding class
    """

    def __init__(self, options):
        assert (isinstance(options, Option))
        self._options = options
        self._session = None
        self.saver = None
        self._cost = None
        self._optimizer = None
        self._word_embeddings = None
        self._para_embeddings = None
        self.vocab = None
        self.vocab_size = 0
        self.document_size = 0
        self.__inputs, self.__labels, self.__lr = None, None, None
        self.__cost = None
        self.__optimizer = None
        self.__summary = None
        self.__normalized_word_embeddings = None

    def setVocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        return self

    def setDocSize(self, doc_size):
        assert (isinstance(doc_size, int))
        self.document_size = doc_size
        return self

    def useSubSampling(self, switch=True, threshold=1e-5):
        self.use_sub_sampling = switch
        self.sub_sampling_threshold = 1e-5
        return self

    def _get_batches(self, doc_ids, word_ids):
        opts = self._options
        return generate_batch_doc2VecC_tail(doc_ids, word_ids, doc_ids, opts.batch_size, opts.window_size)

    def _get_inputs(self):
        """
        Create TF Placeholders for input, targets, and learning rate.
         :return: Tuple of Placeholders (input, targets, learning rate)
        """
        opts = self._options
        inputs_ = tf.placeholder(tf.int32, [opts.batch_size, opts.window_size], name = 'input')
        doc_inputs_ = tf.placeholder(tf.int32, [opts.batch_size, None], name='doc_input')
        labels_ = tf.placeholder(tf.int32, [opts.batch_size, 1], name='label')
        num_sampled_ele_ = tf.placeholder(tf.int32, [opts.batch_size], name='num_sample')
        lr_ = tf.placeholder(tf.float32, name='learning_rate')
        return inputs_, doc_inputs_, labels_, num_sampled_ele_, lr_

    def _get_embedding_layer(self, input_data, doc_input_data, num_doc_sampled_data):
        """
        Create embedding for <input_data> and <doc_input_data>.
        :param input_data: TF placeholder for text input.
        :return: Embedded input tensor.
        """
        opts = self._options
        word_embedding = tf.Variable(tf.random_uniform((self.vocab_size, opts.embed_dim), -1.0, 1.0))
        embed = []

        embed_d = tf.zeros([opts.batch_size, opts.embed_dim])
        for m in range(opts.batch_size):
            for n in range(opts.max_sentence_sample):
                if (doc_input_data[m, n] != 0):
                    embed_d[m, :] += tf.nn.embedding_lookup(word_embedding, doc_input_data[m, n])
            embed_d[m, :] /= num_doc_sampled_data[m]

        if opts.concat == 'True':
            combined_embed_vector_length = opts.embed_dim * opts.window_size + opts.embed_dim
            for j in range(opts.window_size):
                embed_w = tf.nn.embedding_lookup(word_embedding, input_data[:, j])
                embed.append(embed_w)
            embed.append(embed_d)
        else:
            combined_embed_vector_length = opts.embed_dim
            embed_w = tf.zeros([opts.batch_size, opts.embed_dim])
            for j in range(opts.window_size):
                embed_w += tf.nn.embedding_lookup(word_embedding, input_data[:, j])
            embed_w += embed_d
            embed.append(embed_w)

        return tf.concat(embed, 1), word_embedding, combined_embed_vector_length

    def build_graph(self):
        """
        Create Graph and Initialize tf Session for training
        """
        train_graph = tf.Graph()
        opts = self._options
        with train_graph.as_default():
            self.__inputs, self.__doc_inputs, self.__labels, self.__num_sampled_ele, self.__lr = self._get_inputs()
            embed, word_embeddings, combined_embed_vector_length = self._get_embedding_layer(
                self.__inputs, self.__doc_inputs, self.__num_sampled_ele)

            norm_w = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
            self.__normalized_word_embeddings = word_embeddings / norm_w

            weights = tf.Variable(
                tf.truncated_normal((self.vocab_size, combined_embed_vector_length),
                                    stddev=1.0 / math.sqrt(combined_embed_vector_length))
            )
            biases = tf.Variable(tf.zeros(self.vocab_size))

            if opts.loss == 'softmax':
                loss = tf.nn.sampled_softmax_loss(weights=weights,
                                                  biases=biases,
                                                  labels=self.__labels,
                                                  inputs=embed,
                                                  num_sampled=opts.negative_sample_size,
                                                  num_classes=opts.vocab_size)
                tf.summary.scalar("Softmax loss", loss)
            else:
                loss = tf.nn.nce_loss(weights=weights,
                                      biases=biases,
                                      labels=self.__labels,
                                      inputs=embed,
                                      num_sampled=opts.negative_sample_size,
                                      num_classes=opts.vocab_size)
                tf.summary.scalar("NCE loss", loss)

            self.__cost = tf.reduce_mean(loss)

            if opts.train_method == 'Adam':
                self.__optimizer = tf.train.AdamOptimizer(self.__lr).minimize(self.__cost)
            else:
                self.__optimizer = tf.train.GradientDescentOptimizer(self.__lr).minimize(self.__cost)

            self.__summary = tf.summary.merge_all()

            self._session = tf.Session(graph=train_graph)
            self.saver = tf.train.Saver()
        return self

    def fit(self, docs):
        opts = self._options
        iteration = 1
        loss = 0
        doc_ids = [[i] * len(j) for i, j in enumerate(docs)]
        doc_ids = [item for sublist in doc_ids for item in sublist]
        doc_lens = [0] + [len(i) for i in docs]
        for i in range(1, len(doc_lens)):
            doc_lens[i] += doc_lens[i-1]
        word_ids = [item for sublist in docs for item in sublist]

        with self._session as session:
            session.run(tf.global_variables_initializer())
            for e in range(1, opts.epochs_to_train + 11):
                batches = self._get_batches(doc_ids, word_ids, doc_lens)
                start = time.time()
                lr = opts.learning_rate if e <= opts.epochs_to_train else opts.learning_rate * (
                    e - opts.epochs_to_train / 10)
                for x, y, m, n in batches:
                    feed = {self.__inputs: x,
                            self.__labels: y,
                            self.__doc_inputs: m,
                            self.__num_sampled_ele: n,
                            self.__lr: lr}
                    train_loss, _ = session.run([self.__cost, self.__optimizer], feed_dict=feed)

                    loss += train_loss
                    if iteration % opts.statistics_interval == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(e, opts.epochs_to_train + 11),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss * 1.0 / opts.statistics_interval),
                              "{:.4f} sec/batch".format((end - start) * 1.0 / opts.statistics_interval))
                        loss = 0
                        start = time.time()
                    if iteration % opts.checkpoint_interval == 0:
                        self.saver.save(self._session,
                                        "para2vec",
                                        global_step=iteration)
                    iteration += 1
            self._word_embeddings = self.__normalized_word_embeddings.eval()
            self.saver(self._session, "final_para2vec")

    def transform_w(self, word_index):
        return self._word_embeddings[word_index, :]

    def transform_doc(self, word_indexs):
        doc_embeddings = [self._word_embeddings[i, :] for i in word_indexs]
        return doc_embeddings