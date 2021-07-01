import numpy as np
from .helpers import SquaredExponential

class Klr(object):
    def __init__(self, kernel=None):
        self._init_kernel(kernel)

    def _init_kernel(self, kernel):
        if kernel is None:
            #kernel = lambda x1, x2: np.exp(\
            #        -sp.distance.cdist(x1, x2, 'sqeuclidean')/2)
            kernel = SquaredExponential(1)
        self.kernel = kernel

    def _step(self, x, y, lamb):
        Ka = self.K@self.a
        p = 1/(1+np.exp(-Ka)).reshape((-1,))
        W = np.diag(p*(1-p))
        z = self._z(y, W, p, Ka)
        
        KW = self.K.T@W
        self.a = np.linalg.solve(\
                KW@self.K+lamb*np.eye(self.K.shape[0]),
                KW@z)

    def _p_x(self, x):
        return 1/(1+np.exp(-self.K @ self.a)).reshape((-1,))

    def _z(self, y, W, p, Ka):
        return Ka + np.linalg.solve(W, y - p.reshape((-1,1)))

    def fit(self, x, y, lamb = 0.1, num_iters = 10):
        # Initialise
        self.x = x.copy()
        self.a = np.zeros((x.shape[0], 1))
        self.K = self.kernel(x, x)

        # Loop
        for k in range(0, num_iters):
            a_old = self.a.copy()
            self._step(x, y, lamb)
            print(np.linalg.norm(self.a-a_old))
        
        return self.a

    def predict(self, x):
        ker = self.kernel(x, self.x)
        return 1/(1+np.exp(-ker@self.a)), ker@self.a > 0




