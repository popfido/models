# coding=utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as func
from option import BasicOption
import numpy as np
import time


class LinearModuler(nn.Module):
    def __init__(self, num_feature, num_class):
        super(LinearModuler, self).__init__()
        self.linear = nn.Linear(num_feature, num_class)

    def forward(self, x):
        return self.linear(x)


class LogisticModuler(nn.Module):
    def __init__(self, num_feature, num_class):
        super(LogisticModuler, self).__init__()
        self.logistic = nn.Linear(num_feature, num_class)

    def forward(self, x):
        return func.softmax(self.logistic(x))


class BasicClassifier(object):
    def __init__(self, options):
        assert isinstance(options, BasicOption)
        self.option = options
        self.loss = nn.CrossEntropyLoss()
        if option.model == "linear":
            self.model = LinearModuler(options.num_feature, options.num_class)
        elif option.model == "logistic":
            self.model = LogisticModuler(options.num_feature, options.num_class)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=options.lr)

    def fit(self, x_train, y_train):
        loss = 0
        if not isinstance(x_train, torch.FloatTensor):
            x_train = torch.from_numpy(x_train)
        if not isinstance(y_train, torch.FloatTensor):
            y_train = torch.from_numpy(y_train)
        start = time.time()
        for epoch in range(self.option.num_epochs):
            inputs = Variable(x_train)
            labels = Variable(y_train)
            out = self.model(inputs)
            train_loss = self.loss(out, labels)

            self.optimizer.zero_grad()
            train_loss.backward()
            if torch.cuda.is_available():
                loss += train_loss.cpu().data[0]
            else:
                loss += train_loss.data[0]
            self.optimizer.step()

            if (epoch + 1) % self.option.statistics_interval == 0:
                end = time.time()
                print('Epoch[{}/{}], Avg. Training loss: {:.4f}'.format(epoch + 1,
                                                                        self.option.num_epochs,
                                                                        loss / self.option.statistics_interval),
                      "{:.4f} sec/batch".format((end - start) * 1.0 / self.option.statistics_interval))
                loss = 0
                start = time.time()


class BasicRegression(object):
    def __init__(self, options):
        assert isinstance(options, BasicOption)
        self.option = options
        self.loss = nn.MSELoss()
        if option.model == "linear":
            self.model = LinearModuler(options.num_feature, 1)
        elif option.model == "logistic":
            self.model = LogisticModuler(options.num_feature, 1)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=options.lr)

    def fit(self, x_train, y_train):
        loss = 0
        if not isinstance(x_train, torch.FloatTensor):
            x_train = torch.from_numpy(x_train)
        if not isinstance(y_train, torch.FloatTensor):
            y_train = torch.from_numpy(y_train)
        start = time.time()
        for epoch in range(self.option.num_epochs):
            inputs = Variable(x_train)
            labels = Variable(y_train)
            out = self.model(inputs)
            train_loss = self.loss(out, labels)

            self.optimizer.zero_grad()
            train_loss.backward()
            if torch.cuda.is_available():
                loss += train_loss.cpu().data[0]
            else:
                loss += train_loss.data[0]
            self.optimizer.step()

            if (epoch + 1) % self.option.statistics_interval == 0:
                end = time.time()
                print('Epoch[{}/{}], Avg. Training loss: {:.4f}'.format(epoch + 1,
                                                                        self.option.num_epochs,
                                                                        loss / self.option.statistics_interval),
                      "{:.4f} sec/batch".format((end - start) * 1.0 / self.option.statistics_interval))
                loss = 0
                start = time.time()


if __name__ == "__main__":
    option = BasicOption('logistic', 1, 1, 0.001, 1000, 20)
    model = BasicRegression(option)
    x = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    model.fit(x, y)
