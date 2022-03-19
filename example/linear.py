import logging
import sklearn

import numpy as np

from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import OneHotEncoder

from model.linear import LinearRegression, Binaryclassification, MultiClassification
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.ERROR)


def regression1():
    """
    ランダムなregressionのサンプルを生成
    """
    boston = load_boston()
    x_org, yt = boston.data, boston.target
    feature_names = boston.feature_names
    print("Original name", x_org.shape, yt.shape)
    print("Feature name", feature_names)

    x_data = x_org[:, feature_names == "RM"]
    print(x_data.shape)

    x = np.insert(x_data, 0, 1.0, axis=1)
    print(x.shape)

    print(x[:5], yt[:5])

    # model定義
    M = x.shape[0]
    D = x.shape[1]

    model = LinearRegression(D)
    # model学習
    model.fit(iters=10000, alpha=0.01, x=x, yt=yt)
    # model推論


def regression2():
    """
    重回帰サンプル
    """
    boston = load_boston()
    x_org, yt = boston.data, boston.target
    feature_names = boston.feature_names
    print("Original name", x_org.shape, yt.shape)
    print("Feature name", feature_names)

    print(feature_names == "RM")
    print(feature_names == "LSTAT")

    print(np.where((feature_names == "RM") | (feature_names == "LSTAT")))

    feature_rows = np.where((feature_names == "RM") |
                            (feature_names == "LSTAT"))
    x2 = x_org[:, feature_rows[0]]
    print(x2.shape)
    print(x2)
    M, D = x2.shape
    model2 = LinearRegression(D)
    model2.fit(iters=10000, alpha=0.001, x=x2, yt=yt)


def classification():
    """
    分類のサンプルを生成
    """
    iris = load_iris()
    x_org: np.ndarray = iris.data
    y_org: np.ndarray = iris.target
    print("Origin Data", x_org.shape, y_org.shape)

    x_data: np.ndarray = iris.data[:100, :2]
    x_data = np.insert(x_data, 0, 1.0, axis=1)
    y_data: np.ndarray = iris.target[:100]
    print("Data shape", x_data.shape, y_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=70, test_size=30, random_state=123
    )
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    D = x_data.shape[1]
    classification = Binaryclassification(D)

    classification.fit(iters=10000, alpha=0.0001, x=x_train,
                       yt=y_train, x_test=x_test, yt_test=y_test)


def multi_classification():
    """
    多値分類タスク
    """
    iris = load_iris()
    x_org, y_org = iris.data, iris.target
    x_select = x_org[:, [0, 2]]
    x_all = np.insert(x_select, 0, 1.0, axis=1)

    ohe = OneHotEncoder(sparse=False, categories="auto")
    y_work = np.c_[y_org]
    y_all_onehot: np.ndarray = ohe.fit_transform(y_work)

    print(x_all.shape, y_all_onehot.shape)

    x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
        x_all, y_org, y_all_onehot, train_size=75, test_size=75, random_state=123)

    print(x_train.shape, x_test.shape, y_train.shape,
          y_test.shape, y_train_one.shape, y_test_one.shape)

    D = x_all.shape[1]
    N = y_all_onehot.shape[1]
    multiClassModel = MultiClassification(D=D, N=N)

    multiClassModel.fit(iters=50000, alpha=0.001,
                        x=x_train, yt=y_train_one, x_test=x_test, yt_test=y_test, yt_test_onehost=y_test_one)


if __name__ == "__main__":
    # regression1()
    # regression2()
    # classification()
    multi_classification()
