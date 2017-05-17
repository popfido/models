import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from option import BasicOption
import numpy as np
import sys
import time
from config import config_holder


class LinearRegressionModuler(nn.Module):
    def __init__(self, num_feature, num_class):
        super(LinearRegressionModuler, self).__init__()
        self.linear = nn.Linear(num_feature, num_class)

    def forward(self, x):
        return self.linear(x)

class LosgisticRegressionModuler(nn.Module):
    def __init__(self, num_feature, num_class):
        super(LosgisticRegressionModuler, self).__init__()
        self.logistic = nn.Linear(num_feature, num_class)

    def forward(self, x):
        return F.softmax(nn.self.logistic(x))

class LinearRegression(object):
    def __init__(self, options):
        assert isinstance(options, BasicOption)
        self.option = options
        self.loss = nn.MSELoss()
        if option.model == "linear":
            self.model = LinearRegressionModuler(options.num_feature, options.num_class)
        elif option.model == "logistic":
            self.model = LosgisticRegressionModuler(options.num_feature, options.num_class)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=options.lr)

    def fit(self, x_train, y_train, num_epochs=1000):
        loss = 0
        if not isinstance(x_train, torch.FloatTensor):
            x_train = torch.from_numpy(x_train)
        if not isinstance(y_train, torch.FloatTensor):
            y_train = torch.from_numpy(y_train)
        start = time.time()
        for epoch in range(num_epochs):
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
                                                          num_epochs,
                                                          loss/self.option.statistics_interval),
                      "{:.4f} sec/batch".format((end - start) * 1.0 / self.option.statistics_interval))
                loss = 0
                start = time.time()

if __name__ == "__main__":
    sys.argv = ['-train'] + ['./train.txt'] + ['-stat_interval'] + ['20']
    config_holder.init()
    option = BasicOption('logistic', 1, 1, 0.001, 20)
    model = LinearRegression(option)
    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

    model.fit(x_train, y_train)