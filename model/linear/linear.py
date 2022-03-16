import numpy as np


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
