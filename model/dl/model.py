import numpy as np
from sklearn.metrics import accuracy_score
from model.utils import multi_cross_entropy


class MLP(object):
    """
    三層構造のMLP
    """

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        推論
        """
        pass

    def evaluate(self, x_test: np.ndarray, yt_test: np.ndarray, yt_test_onehot: np.ndarray):
        """
        予測結果の評価
        """
        yp_test_onehot = self.predict(x_test)
        ce_loss = multi_cross_entropy(yt_test_onehot, yp_test_onehot)

        yp_test = np.argmax(yp_test_onehot, axis=1)
        score = accuracy_score(yt_test, yp_test)

        return score, ce_loss
