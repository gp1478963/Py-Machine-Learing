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

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        print(X.shape)
        self.cost_ = []
        print(X.T)
        for i in range(self.n_iter):
            # 计算all targets
            num_net_inputs = self.net_input(X)
            # 走激活函数
            post_actives = self.activaty(num_net_inputs)
            # 计算残差
            errors = (y - post_actives)
            # loss偏导
            mse = X.T.dot(errors)
            # 更新权重
            self.coef_[1:] = self.coef_[1:] + self.learn_rate * mse
            self.coef_[0] += self.learn_rate * errors.sum()
            # 计算loss
            loss = (errors ** 2).sum()  / 2.0
            self.cost_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.coef_[1:]) + self.coef_[0]

    def activaty(self, X):
        return X

    def predict(self, X):
        return np.where(self.activaty(self.net_input(X)) >= 0.0, 1, -1)


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

adaline = Adaline(n_iter=20, random_state=1, learn_rate=0.0001)
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