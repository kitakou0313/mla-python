import logging

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification, make_regression

logging.basicConfig(level=logging.ERROR)


def regression():
    """
    ランダムなregressionのサンプルを生成
    """
    X, y = make_regression(
        n_samples=10000, n_features=100, n_informative=75, n_targets=1, noise=0.05, random_state=111, bias=0.5
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=111
    )

    print(X[:10])
    print(y[:10])

    # model定義
    # model推論


def classification():
    """
    分類のサンプルを生成
    """
    X, y = make_classification(
        n_samples=1000, n_features=100, n_informative=75, random_state=111, n_classes=2, class_sep=2.5
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=111
    )

    print(X[:10])
    print(y[:10])


if __name__ == "__main__":
    regression()
    classification()
