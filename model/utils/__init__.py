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
