from matplotlib.pyplot import hist
from sklearn.metrics import accuracy_score
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """
    シグモイド
    """
    return 1 / (1 + jnp.exp(-x))


sigmoid_jit = jit(sigmoid)


def classify(y):
    return jnp.where(y < 0.5, 0, 1)


def cross_entropy(W: jnp.ndarray, b: jnp.ndarray,  x_train: jnp.ndarray, yt: jnp.ndarray) -> jnp.ndarray:
    """
    損失関数(cross entropy)
    """
    yp = sigmoid(x_train @ W + b)

    ce = yt * jnp.log(yp) + (1-yt) * jnp.log(1-yp)
    return -jnp.sum(ce)


class JaxBinClassification(object):
    """
    Jax version bi classification
    """

    def __init__(self, D: int, key):
        """
        コンストラクタ
        """
        key, W_key, b_key = random.split(key, 3)
        self.W = random.normal(W_key, (D,))
        self.b = random.normal(b_key, (1,))

    def pred(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        推論
        """
        return sigmoid_jit(x @ self.W + self.b)

    def fit(self, alpha: int, iters: int, x_train: jnp.ndarray, yt_train: jnp.ndarray, x_test: jnp.ndarray, yt_test: jnp.ndarray):
        """
        学習
        """
        M = x_train.shape[0]

        loss_W_grad = grad(cross_entropy, argnums=0)
        loss_W_grad = jit(loss_W_grad)

        loss_b_grad = grad(cross_entropy, argnums=1)
        loss_b_grad = jit(loss_b_grad)

        history = []
        for k in range(1, iters+1):
            self.W -= (alpha / M) * \
                (loss_W_grad(self.W, self.b,  x_train, yt_train))

            self.b -= (alpha / M) * \
                (loss_b_grad(self.W, self.b, x_train, yt_train))

            if k % 100 == 0:
                loss, score = self.evaluate(x_test=x_test, yt=yt_test)

                print(self.W)

                history.append((loss, score))
                print("iter", k, loss, score)

        print(history[0])
        print(history[-1])

    def evaluate(self, yt: jnp.ndarray, x_test: jnp.ndarray):
        """
        分類の評価関数
        """
        loss = cross_entropy(self.W, self.b, x_train=x_test, yt=yt)

        yp = self.pred(x_test)
        yp_b = classify(yp)
        score = accuracy_score(yt, yp_b)
        return loss, score


def softmax(x: jnp.ndarray) -> jnp.ndarray:
    """
    ソフトマックス
    """
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = jnp.exp(x)

    return (w / w.sum(axis=0)).T


softmax_jit = jit(softmax)


def multi_cross_entropy(W: jnp.ndarray, x: jnp.ndarray, yt_onehot: jnp.ndarray) -> jnp.ndarray:
    """
    損失関数(多クラス用cross entropy)
    """
    yp = softmax_jit(x @ W)
    return -jnp.sum(jnp.sum(yt_onehot * jnp.log(yp), axis=1))


multi_cross_entropy_jit = jit(multi_cross_entropy)


class JaxMultiClassification(object):
    """
    Jax version multi classification
    """

    def __init__(self, D: int, N: int, key):
        """
        コンストラクタ
        D...特徴
        N...分類クラス
        """
        key, W_key = random.split(key, 2)
        self.W = random.normal(W_key, (D, N))

    def pred(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        推論
        """
        return softmax_jit(x @ self.W)

    def fit(self, alpha: int, iters: int, x_train: jnp.ndarray, yt_train_onehot: jnp.ndarray, x_test: jnp.ndarray, yt_test: jnp.ndarray, yt_test_onehot: jnp.ndarray):
        """
        学習
        """
        M = x_train.shape[0]

        loss_W_grad = grad(multi_cross_entropy_jit, argnums=0)
        loss_W_grad = jit(loss_W_grad)

        history = []
        for k in range(1, iters+1):
            self.W -= (alpha / M) * \
                (loss_W_grad(self.W, x_train, yt_train_onehot))

            if k % 100 == 0:
                loss, score = self.evaluate(
                    x_test=x_test, yt=yt_test, yt_onehot=yt_test_onehot)
                history.append((loss, score))
                print("iter", k, loss, score)

        print(history[0])
        print(history[-1])

    def evaluate(self, yt: jnp.ndarray, x_test: jnp.ndarray, yt_onehot: jnp.ndarray):
        """
        分類の評価関数
        """
        loss = multi_cross_entropy_jit(self.W, x=x_test, yt_onehot=yt_onehot)

        yp = self.pred(x_test)
        yp_test = jnp.argmax(yp, axis=1)

        score = accuracy_score(yt, yp_test)
        return loss, score
