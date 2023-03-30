import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, [1, 2]]
y = iris.target


def load_iris_data_set():
    print(X)
    print(y)
    print(np.unique(y))


# load_iris_data_set()


def test_split():
    # stratify=y: 输入时y什么比例，那么输出时y就什么比例
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)
    print(np.bincount(y_train))
    print(np.bincount(y_test))
    # output
    # [35 35 35]
    # [15 15 15]
    # 当不设置y的比例设定时
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    print(np.bincount(y_train))
    print(np.bincount(y_test))
    # output
    # [36 32 37]
    # [14 18 13]


# test_split()


def normal_dataset():
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)

    X_train_std = standard_scaler.transform(X_train)
    X_test_std = standard_scaler.transform(X_test)
    print(X_train_std)


# normal_dataset()


def try_sklearn_perceptron():
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, stratify=y)
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)

    X_train_std = standard_scaler.transform(X_train)
    X_test_std = standard_scaler.transform(X_test)

    # eta0:学习速率
    perceptron = Perceptron(eta0=0.01, random_state=1)
    perceptron.fit(X_train_std, y_train)

    y_predict = perceptron.predict(X_test_std)
    print('错误个数:', (y_predict != y_test).sum())
    print("预测准确率:", perceptron.score(X_test_std, y_test))
    print("预测准确率:", accuracy_score(y_test, y_predict))
    for y1, y2 in zip(y_test, y_predict):
        print(y1, '\t', y2)


try_sklearn_perceptron()
