from matplotlib.pyplot import hist
import numpy as np
from sklearn.metrics import accuracy_score
from model.utils import multi_cross_entropy, softmax, sigmoid, ReLU, step


class MLP(object):
    """
    三層構造のMLP
    """

    def __init__(self, D: int, N: int, middle_dim: int):
        """
        コンストラクタ
        D...特徴次元数
        N...分類クラス数
        middle_mid...第一層通過後の特徴ベクトル次元数
        """
        # 1層目
        self.V = np.random.randn(D, middle_dim) / np.sqrt(D/2)
        # 2層目（bias項が追加されるので+1）
        self.W = np.random.randn(middle_dim + 1, N) / \
            np.sqrt(middle_dim + 1 / 2)

        # 中間層での特徴ベクトル
        self.b = np.zeros(middle_dim)
        self.b_bias = np.zeros(middle_dim + 1)

        self.a = np.zeros(middle_dim)

    def fit(self, alpha: np.float64, iters: int, x_train: np.ndarray, yt_train: np.ndarray, yt_train_onehot: np.ndarray, x_test: np.ndarray, yt_test: np.ndarray, yt_test_onehot: np.ndarray):
        """
        学習
        """
        history = []

        M = x_train.shape[0]

        for k in range(1, iters+1):
            yp = self.predict(x_train)
            # 誤差計算
            # 予測値誤差
            yd = yp - yt_train_onehot
            # 隠れ層誤差
            # bd = self.b * (1 - self.b) * (yd @ self.W[1:].T)
            bd = step(self.a) * (yd @ self.W[1:].T)

            self.W = self.W - (alpha / M)*(self.b_bias.T @ yd)
            self.V = self.V - (alpha / M)*(x_train.T @ bd)

            if k % 20 == 0:
                score, loss = self.evaluate(
                    x_test=x_test, yt_test=yt_test, yt_test_onehot=yt_test_onehot)
                history.append((score, loss))

                print((score, loss))

        print("学習初期", history[0])
        print("学習終了時", history[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推論
        """
        # 1層目
        a = x@self.V
        self.a = a
        self.b = ReLU(a)

        b_with_bias = np.insert(self.b, 0, 1, axis=1)
        self.b_bias = b_with_bias

        # 2層目
        u = b_with_bias @ self.W
        yp = softmax(u)

        return yp

    def evaluate(self, x_test: np.ndarray, yt_test: np.ndarray, yt_test_onehot: np.ndarray):
        """
        予測結果の評価
        """
        yp_test_onehot = self.predict(x_test)
        ce_loss = multi_cross_entropy(yt_test_onehot, yp_test_onehot)

        yp_test = np.argmax(yp_test_onehot, axis=1)
        score = accuracy_score(yt_test, yp_test)

        return score, ce_loss
