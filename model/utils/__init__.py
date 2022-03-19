import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    シグモイド関数
    """
    return 1 / (1 + np.exp(-x))


def cross_entropy(yt: np.ndarray, yp: np.ndarray) -> np.float64:
    """
    クロスエントロピー
    """
    ce = -(yt * np.log(yp) + (1-yt)*(1-yp))
    return np.mean(ce)


def multi_cross_entropy(yt: np.ndarray, yp: np.ndarray) -> np.float64:
    """
    多クラス分類用のクロスエントロピー
    """
    return -np.mean(np.sum(yt * np.log(yp), axis=1))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    ソフトマックス
    """
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)

    return (w / w.sum(axis=0)).T
