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


def cross_entropy(W: jnp.ndarray, x_train: jnp.ndarray, yt: jnp.ndarray) -> jnp.ndarray:
    """
    損失関数(cross entropy)
    """
    yp = sigmoid(x_train @ W)

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
        key, W_key = random.split(key, 2)
        self.W = random.normal(W_key, (D,))

    def pred(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        推論
        """
        return sigmoid_jit(x @ self.W)

    def fit(self, alpha: int, iters: int, x_train: jnp.ndarray, yt_train: jnp.ndarray, x_test: jnp.ndarray, yt_test: jnp.ndarray):
        """
        学習
        """
        M = x_train.shape[0]

        loss_W_grad = grad(cross_entropy, argnums=0)
        loss_W_grad = jit(loss_W_grad)

        history = []
        for k in range(1, iters+1):
            self.W -= (alpha / M)*(loss_W_grad(self.W, x_train, yt_train))

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
        loss = cross_entropy(self.W, x_train=x_test, yt=yt)

        yp = self.pred(x_test)
        yp_b = classify(yp)
        score = accuracy_score(yt, yp_b)
        return loss, score
