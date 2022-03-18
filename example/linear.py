from json import load
import logging
from matplotlib.pyplot import axis
import sklearn

import numpy as np

from sklearn.datasets import load_boston
from sklearn.utils import Bunch

from model.linear import LinearRegression

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


# def classification():
#     """
#     分類のサンプルを生成
#     """
#     X, y = make_classification(
#         n_samples=1000, n_features=100, n_informative=75, random_state=111, n_classes=2, class_sep=2.5
#     )
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.1, random_state=111
#     )

#     print(X[:10])
#     print(y[:10])


if __name__ == "__main__":
    # regression1()
    regression2()
    # classification()
