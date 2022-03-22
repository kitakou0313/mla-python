import torchvision
import numpy as np

from sklearn.preprocessing import OneHotEncoder


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
    train = np.insert(train, 0, 1, axis=1)
    train = train / 255.0
    train_target: np.ndarray = train_tensor.targets.numpy()

    ohe = OneHotEncoder(sparse=False)
    train_target_onehot = ohe.fit_transform(np.c_[train_target])
    print('One Hot Vector化後', train_target_onehot.shape)

    test: np.ndarray = test_tensor.data.numpy()
    test = test.reshape([test.shape[0], -1])
    test = np.insert(test, 0, 1, axis=1)
    test = test / 255.0
    test_target: np.ndarray = test_tensor.targets.numpy()
    test_target_onehot = ohe.transform(np.c_[test_target])
    print("One hot vector ", test_target_onehot)

    pass


if __name__ == "__main__":
    classificate_mnist()
