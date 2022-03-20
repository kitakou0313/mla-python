import torchvision
import numpy as np


def classificate_mnist():
    """
    MNISTの分類
    """
    train_tensor = torchvision.datasets.MNIST(
        root="/workspaces/mla-python/example/data", train=True, download=True)
    test_tensor = torchvision.datasets.MNIST(
        root="/workspaces/mla-python/example/data", train=False, download=True)

    train: np.ndarray = train_tensor.data.numpy()
    train = train.reshape([train.shape[0], -1])
    train_target: np.ndarray = train_tensor.targets.numpy()
    test: np.ndarray = test_tensor.data.numpy()
    test = test.reshape([test.shape[0], -1])
    test_taeget: np.ndarray = test_tensor.targets.numpy()

    pass


if __name__ == "__main__":
    classificate_mnist()
