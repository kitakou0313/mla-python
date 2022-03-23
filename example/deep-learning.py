from statistics import mode
import torchvision
import numpy as np

from sklearn.preprocessing import OneHotEncoder

from model.dl import MLP


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
    train_target_onehot: np.ndarray = ohe.fit_transform(np.c_[train_target])
    print('One Hot Vector化後', train_target_onehot.shape)

    train = train[:10000, :]
    train_target_onehot = train_target_onehot[:10000, :]

    test: np.ndarray = test_tensor.data.numpy()
    test = test.reshape([test.shape[0], -1])
    test = np.insert(test, 0, 1, axis=1)
    test = test / 255.0
    test_target: np.ndarray = test_tensor.targets.numpy()
    test_target_onehot: np.ndarray = ohe.transform(np.c_[test_target])
    print("One hot vector ", test_target_onehot)

    D = train.shape[1]
    N = train_target_onehot.shape[1]

    model = MLP(D=D, N=N, middle_dim=20)

    model.fit(alpha=0.01, iters=1000, x_train=train,
              yt_train=train_target, yt_train_onehot=train_target_onehot, x_test=test, yt_test=test_target, yt_test_onehot=test_target_onehot)


if __name__ == "__main__":
    np.random.seed(11111)
    classificate_mnist()
