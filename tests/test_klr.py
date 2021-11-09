import pytest

import numpy as np
from klr.helpers import SquaredExponential
from klr.klr import Klr


X1 = np.random.normal(-1, 1, (100,2))
X2 = np.random.normal(1, 1, (201,2))
Y1 = np.zeros((X1.shape[0],1))
Y2 = np.ones((X2.shape[0],1))

X = np.vstack((X1, X2))
kernel_func = SquaredExponential(1)
K = kernel_func(X,X)

Y = np.vstack((Y1, Y2))

train_indices = list(range(80)) + list(range(100,260))
test_indices = list(range(80,101)) + list(range(260,301))

K_train = K[np.ix_(train_indices, train_indices)]
y_train = Y[np.ix_(train_indices,[0])]
X_train = X[np.ix_(train_indices,[0,1])]

K_test = K[np.ix_(test_indices, train_indices)]
y_test = Y[np.ix_(test_indices,[0])]
X_test = X[np.ix_(test_indices,[0,1])]

def test_Klr():
    # feature vector input
    model = Klr(None, precomputed_kernel=False)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # kernel matrix input
    model = Klr(None, precomputed_kernel=True)
    model.fit(K_train, y_train)
    y_pred_kernel = model.predict(K_test)

    print(y_pred.flatten())
    print(y_pred_kernel.flatten())
    assert y_pred.shape == y_test.shape
    assert y_pred_kernel.shape == y_test.shape
    assert list(y_pred.flatten()) == list(y_pred_kernel.flatten())