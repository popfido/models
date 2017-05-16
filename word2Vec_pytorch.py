"""

Tensorflow implementation of WV-DR algorithm wrapper class(sklearn style)

:authro: Fido Wang (wanghailin317@gmail.com)
:refer: http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from option import Option
import math
import time
import collections
from itertools import compress
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

    i=0
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
            yield batch, labels[:,None]
            i = 0
            labels = np.ndarray(shape=(batch_size), dtype=np.int32)
            batch = np.ndarray(shape=(batch_size), dtype=np.int32)


class SkipgramModeler(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipgramModeler, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.out_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.out_embed.weight = nn.Parameter(torch.FloatTensor(self.vocab_size, self.embed_dim).uniform_(-1, 1))

        self.input_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.input_embed.weight = nn.Parameter(torch.FloatTensor(self.vocab_size, self.embed_dim).uniform_(-1, 1))

    def forward(self, inputs, labels, num_sampled):
        use_cuda = self.out_embed.weight.is_cuda

        [batch_size, window_size] = labels.size()

        input = self.input_embed(inputs.repeat(1, window_size).contiguous().view(-1))
        output = self.out_embed(labels.view(-1))

        noise = nn. Variable(torch.Tensor(batch_size * window_size, num_sampled).uniform_(0, self.vocab_size - 1).long())
        if use_cuda:
            noise = noise.cuda()
        noise = self.out_embed(noise).neg()

        log_target = (input * output).sum(1).squeeze().sigmoid().log()

        ''' 
        ∑[batch_size * window_size, num_sampled, embed_size] * [batch_size * window_size, embed_size, 1] ->
        ∑[batch_size, num_sampled, 1] -> [batch_size] 
        '''
        sum_log_sampled = torch.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        loss = log_target + sum_log_sampled

        return -loss.sum() / batch_size

    def get_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


class Skipgram(object):
    """
    Skip-Gram word embedder

    Support Functional Style setting thus all function begin with 'set' and 'build'
    will return an object,the Embedding Estimator itself.
    """
    def __init__(self, options):
        assert(isinstance(options, Option))
        self._options = options
        self._session = None
        self.saver = None
        self.model = SkipgramModeler(options.vocab_size, options.embed_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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

    def _get_batches(self, doc_ids, word_ids, batch_size, skip_size, window_size):
        opts = self._options
        n_batches = len(word_ids)//batch_size
        words = word_ids[:n_batches*batch_size]
        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx+batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = self._get_target(batch, ii, window_size)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            yield np.array(x), np.array(y)[:, None]

    def _get_target(self, words, idx, window_size=5):
        R = np.random.randint(1, window_size+1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = set(words[start:idx] + words[idx+1:stop+1])

        return list(target_words)

    def _choose_batch_generator(self, chosen='words'):
        if chosen == 'words':
            self._generate_batches = self._get_batches
        if chosen == 'docs':
            self._generate_batches = generate_batch_para

    def fit(self, train_data):
        opts = self._options
        iteration = 1
        loss = torch.Tensor([0])

        if type(train_data[0]) is int:
            word_ids = train_data
            doc_ids = [0] * len(train_data)
            self._choose_batch_generator(chosen='words')
        elif type(train_data[0]) is list:
            doc_ids = [[i]*len(j) for i,j in enumerate(train_data)]
            doc_ids = [item for sublist in doc_ids for item in sublist]
            word_ids = [item for sublist in train_data for item in sublist]
            self._choose_batch_generator(chosen='docs')

        for e in range(1, opts.epochs_to_train+11):
            batches = self._generate_batches(doc_ids, word_ids, opts.batch_size, opts.window_size, opts.window_size)
            start = time.time()
            learning_rate = opts.learning_rate if e <= opts.epochs_to_train else opts.learning_rate * (e-opts.epochs_to_train/10)
            for x, y in batches:
                context_var = autograd.Variable(torch.from_numpy((x)))

                train_loss = self.model(context_var)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                loss += train_loss
                if iteration % opts.statistics_interval == 0:
                    end = time.time()
                    print("Epoch {}/{}".format(e, opts.epochs_to_train + 10),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss/opts.statistics_interval),
                          "{:.4f} sec/batch".format((end-start)*1.0/opts.statistics_interval))
                    loss = 0
                    start = time.time()
                iteration += 1
