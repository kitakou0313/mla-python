from re import M
from matplotlib.pyplot import hist
import numpy as np
from model.utils import sigmoid, cross_entropy, softmax, multi_cross_entropy
from sklearn.metrics import accuracy_score


def classify(y):
    return np.where(y < 0.5, 0, 1)


def evaluate(yt: np.ndarray, yp: np.ndarray):
    """
    分類の評価関数
    """
    loss = cross_entropy(yt, yp)
    yp_b = classify(yp)
    score = accuracy_score(yt, yp_b)

    return loss, score


class Binaryclassification(object):
    """
    2値分類
    """

    def __init__(self, D: int):
        """
        コンストラクタ
        D:int 入力次元
        """
        self.w = np.zeros(D)
        self.D = D

    def pred(self, x: np.ndarray) -> np.ndarray:
        """
        予測
        """
        return sigmoid(x @ self.w)

    def fit(self, iters: int, alpha: np.float64, x: np.ndarray, yt: np.ndarray, x_test: np.ndarray, yt_test: np.ndarray):
        """
        学習
        """
        M, D = x.shape

        history = []

        for k in range(1, iters+1):
            yp = self.pred(x)
            yd = yp - yt

            self.w = self.w - (alpha / M) * (x.T @ yd)
            if k % 100 == 0:
                yp_test = self.pred(x_test)
                loss, score = evaluate(yt_test, yp_test)
                history.append((loss, score))

                print("Loss, score", loss, score, self.w)

        print(history[0])
        print(history[-1])


class MultiClassification(object):
    """
    複数クラス分類
    """

    def __init__(self, D: int, N: int):
        """
        コンストラクタ
        D...特徴次元数
        N...分類クラス数
        """
        self.W = np.ones((D, N))

    def pred(self, x: np.ndarray) -> np.ndarray:
        """
        予測関数
        """
        return softmax(x @ self.W)

    def fit(self, iters: int, alpha: np.float64, x: np.ndarray, yt: np.ndarray, x_test: np.ndarray, yt_test: np.ndarray, yt_test_onehost: np.ndarray):
        """
        学習
        """
        history = []
        for iter in range(1, iters+1):
            M, D = x.shape

            yp = self.pred(x)
            yd = yp - yt

            self.W = self.W - (alpha / M)*(x.T @ yd)

            if iter % 10 == 0:
                loss, score = self.evaluate(
                    x_test=x_test, y_test=yt_test, y_test_one=yt_test_onehost)
                history.append((loss, score))

                print("iter, loss, score", iter, loss, score)

        print(history[0])
        print(history[-1])

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, y_test_one):
        """
        多クラス分類の評価
        """
        yp_test_one = self.pred(x_test)

        yp_test = np.argmax(yp_test_one, axis=1)
        loss = multi_cross_entropy(y_test_one, yp_test_one)
        score = accuracy_score(y_test, yp_test)
        return loss, score


class LinearRegression(object):
    """
    線形回帰
    """

    def __init__(self, D: int):
        """
        コンストラクタ
        D: int 入力次元
        """
        self.w = np.zeros(D)
        self.D = D

    def fit(self, iters: int, alpha: np.float64, x: np.ndarray, yt: np.ndarray):
        """
        学習
        """
        M = x.shape[0]
        D = self.D

        history = np.array([])

        for k in range(1, iters+1):
            yp = self.predict(x)
            yd = yp-yt

            self.w = self.w - alpha / M * (x.T @ yd)

            if (k % 100 == 0):
                # 損失関数値の計算 (7.6.1)
                loss = np.mean(yd ** 2) / 2
                # 計算結果の記録
                history = np.append(history, loss)
                # 画面表示
                print("iter = %d  loss = %f" % (k, loss))

        print("Loss init", history[0])
        print("Final Loss", history[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推論
        """
        return (x @ self.w)
