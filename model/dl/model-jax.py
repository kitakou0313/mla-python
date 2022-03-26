import jax.numpy as jnp
from jax import grad, random, jit


class MLP(object):
    """
    三層構造のMLP
    """

    def __init__(self, D: int, N: int, middle_dim: int, W_key, b_key):
        """
        コンストラクタ
        D...特徴次元数
        N...分類クラス数
        middle_mid...第一層通過後の特徴ベクトル次元数
        """
        # 1層目
        self.V = random.normal(D, middle_dim) / jnp.sqrt(D/2)
        # 2層目（bias項が追加されるので+1）
        self.W = random.normal(middle_dim + 1, N) / \
            jnp.sqrt(middle_dim + 1 / 2)

        # 中間層での特徴ベクトル
        self.b = jnp.zeros(middle_dim)
        self.b_bias = jnp.zeros(middle_dim + 1)

        self.a = jnp.zeros(middle_dim)

    def fit(self, alpha: jnp.float64, iters: int, x_train: jnp.ndarray, yt_train: jnp.ndarray, yt_train_onehot: jnp.ndarray, x_test: jnp.ndarray, yt_test: jnp.ndarray, yt_test_onehot: jnp.ndarray):
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

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        推論
        """
        # 1層目
        a = x@self.V
        self.a = a
        self.b = ReLU(a)

        b_with_bias = jnp.insert(self.b, 0, 1, axis=1)
        self.b_bias = b_with_bias

        # 2層目
        u = b_with_bias @ self.W
        yp = softmax(u)

        return yp

    def evaluate(self, x_test: jnp.ndarray, yt_test: jnp.ndarray, yt_test_onehot: jnp.ndarray):
        """
        予測結果の評価
        """
        yp_test_onehot = self.predict(x_test)
        ce_loss = multi_cross_entropy(yt_test_onehot, yp_test_onehot)

        yp_test = jnp.argmax(yp_test_onehot, axis=1)
        score = accuracy_score(yt_test, yp_test)

        return score, ce_loss
