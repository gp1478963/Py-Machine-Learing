import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


class Adaline(object):
    def __init__(self, n_iter=10, random_state=1, learn_rate=0.1):
        self.n_iter = n_iter
        self.random_state = random_state
        self.learn_rate = learn_rate
        self.coef_ = []
        self.cost_ = []
        self.is_pretreatment = False

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        self.cost_ = []
        X_std = self.pretreatment(X)
        for i in range(self.n_iter):
            # 计算all targets
            num_net_inputs = self.net_input(X_std)
            # 走激活函数
            post_actives = self.activaty(num_net_inputs)
            # 计算残差
            errors = (y - post_actives)
            # loss偏导
            mse = X_std.T.dot(errors)
            # 更新权重
            self.coef_[1:] = self.coef_[1:] + self.learn_rate * mse
            self.coef_[0] += self.learn_rate * errors.sum()
            # 计算loss
            loss = (errors ** 2).sum() / 2.0
            self.cost_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.coef_[1:]) + self.coef_[0]

    def activaty(self, X):
        return X

    def predict(self, X):
        X_std = self.pretreatment(X)
        return np.where(self.activaty(self.net_input(X_std)) >= 0.0, 1, -1)

    def pretreatment(self, X):
        X_std = np.copy(X)
        if self.is_pretreatment:
            for index in range(X_std.shape[-1]):
                X_std[:, index] = (X_std[:, index] - X_std[:, index].mean()) / X_std[:, index].std()
        return X_std


def TestClassAdaline():
    dataset_iris = pd.read_csv('../data/iris.data', header=None, encoding='utf-8')

    dataset_iris_all = dataset_iris.iloc[:100, :].values
    np.random.shuffle(dataset_iris_all)

    print(dataset_iris_all.shape)

    X_train = dataset_iris_all[:70, :-2]
    X_test = dataset_iris_all[70:100, :-2]

    Y_train = np.where(dataset_iris_all[:70, -1] == 'Iris-setosa', 1, -1)
    Y_test = np.where(dataset_iris_all[70:100, -1] == 'Iris-setosa', 1, -1)

    # print(X_train)
    # print('\n', X_test)

    adaline = Adaline(n_iter=5, random_state=1, learn_rate=0.01)
    adaline.is_pretreatment = True
    adaline.fit(X_train, Y_train)
    print(adaline.coef_)
    print(adaline.cost_)

    y_predict = adaline.predict(X_test)
    for y_, y_t in zip(y_predict, Y_test):
        print(y_, '\t', y_t)

    plt.style.use('_mpl-gallery')

    # make data

    # plot
    fig, ax = plt.subplots()
    x = range(len(adaline.cost_))
    y = adaline.cost_
    # print(x)
    # print(y)
    ax.plot(x, y, linewidth=2.0)
    plt.show()


class AdalineSGD(Adaline):
    def __init__(self, n_iter=10, random_state=1, learn_rate=0.1, shuffle=True):
        Adaline.__init__(self, n_iter=n_iter, random_state=random_state, learn_rate=learn_rate)
        self.shuffle = shuffle
        self.w_initialized = False

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.coef_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def fit(self, X, y):
        X_std = self.pretreatment(X)
        self._initialize_weights(X_std.shape[-1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X_std, y = self._shuffle(X, y)
            cost_per = []
            for xi, yi in zip(X_std, y):
                cost_post = self._update_weights(xi, yi)
                cost_per.append(cost_post)
            avg_cost = sum(cost_per) / len(y)
            self.cost_.append(avg_cost)
        return self

    def _update_weights(self, xi, yi):
        post = self.activaty(self.net_input(xi))
        diff = yi - post
        self.coef_[1:] = self.coef_[1:] + self.learn_rate * xi.dot(diff)
        self.coef_[0] += self.learn_rate * diff
        cost = diff**2 / 2.0
        return cost


def TestClassAdalineSGD():
    dataset_iris = pd.read_csv('../data/iris.data', header=None, encoding='utf-8')

    dataset_iris_all = dataset_iris.iloc[:100, :].values
    np.random.shuffle(dataset_iris_all)

    print(dataset_iris_all.shape)

    X_train = dataset_iris_all[:70, :-2]
    X_test = dataset_iris_all[70:100, :-2]

    Y_train = np.where(dataset_iris_all[:70, -1] == 'Iris-setosa', 1, -1)
    Y_test = np.where(dataset_iris_all[70:100, -1] == 'Iris-setosa', 1, -1)

    adaline_sgd = AdalineSGD(n_iter=6, random_state=1, learn_rate=0.0001, shuffle=True)
    adaline_sgd.is_pretreatment = True
    adaline_sgd.fit(X_train, Y_train)
    print(adaline_sgd.coef_)
    print(adaline_sgd.cost_)

    y_predict = adaline_sgd.predict(X_test)
    for y_, y_t in zip(y_predict, Y_test):
        print(y_, '\t', y_t)

    plt.style.use('_mpl-gallery')

    # plot
    fig, ax = plt.subplots()
    x = range(len(adaline_sgd.cost_))
    y = adaline_sgd.cost_
    # print(x)
    # print(y)
    ax.plot(x, y, linewidth=2.0)
    plt.show()


TestClassAdalineSGD()















































