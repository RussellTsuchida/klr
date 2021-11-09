import pytest

import numpy as np
from klr.helpers import SquaredExponential
from klr.ovr import Ovr



X1 = np.random.normal(-1, 1, (100,2))
X2 = np.random.normal(1, 1, (201,2))
X3 = np.random.normal(3, 1, (100,2))
Y1 = np.zeros((X1.shape[0],1))
Y2 = np.ones((X2.shape[0],1))
Y3 = np.repeat(3,X3.shape[0]).reshape(-1,1)

X = np.vstack((X1, X2, X3))
kernel_func = SquaredExponential(1)
K = kernel_func(X,X)

Y = np.vstack((Y1, Y2, Y3))

train_indices = list(range(80)) + list(range(100,260)) + list(range(301,350))
test_indices = list(range(80,101)) + list(range(260,301)) +list(range(350,401))

K_train = K[np.ix_(train_indices, train_indices)]
y_train = Y[np.ix_(train_indices,[0])]
X_train = X[np.ix_(train_indices,[0,1])]

K_test = K[np.ix_(test_indices, train_indices)]
y_test = Y[np.ix_(test_indices,[0])]
X_test = X[np.ix_(test_indices,[0,1])]

def test_Ovr():
    # feature vector input
    # model = Ovr(X_train, y_train, precomputed_kernel=False)
    # y_pred = model.predict(X_test)

    # kernel matrix input
    model = Ovr(K_train, y_train, precomputed_kernel=True)
    y_pred_kernel = model.predict(K_test)

    # print(y_pred.flatten())
    print(y_pred_kernel.flatten())
    print(y_test.flatten())
    # assert y_pred.shape == y_test.shape
    assert y_pred_kernel.shape == y_test.shape
    # assert list(y_pred.flatten()) == list(y_pred_kernel.flatten())