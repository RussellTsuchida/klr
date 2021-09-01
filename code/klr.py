import numpy as np
from helpers import SquaredExponential

class Klr(object):
    """klr implementation based on 
    https://papers.nips.cc/paper/2001/file/2eace51d8f796d04991c831a07059758-Paper.pdf,
    with similar annotation of variables

    Args:
        object ([type]): [description]
    """
    def __init__(self, kernel_func=None, precomputed_kernel=True):
        self.precomputed_kernel = precomputed_kernel
        if not self.precomputed_kernel:
            self._init_kernel_func(kernel_func)

    def _init_kernel_func(self, kernel_func):
        """specify the function to compute kernels

        Args:
            kernel (func ?): kernel function
        """
        if kernel_func is None:
            #kernel = lambda x1, x2: np.exp(\
            #        -sp.distance.cdist(x1, x2, 'sqeuclidean')/2)
            kernel_func = SquaredExponential(1)
        self.kernel_func = kernel_func

    def _step(self, y, lamb):
        """procedures to be repeated in each iteration of optimising prediction

        Args:
            y (np.array): training responses
            lamb: the lambda coefficient
        """
        self.Ka = self.K@self.a
        p = 1/(1+np.exp(-self.Ka)).reshape((-1,))
        W = np.diag(p*(1-p))
        z = self._z(y, W, p, self.Ka)
        
        KW = self.K.T@W
        self.a = np.linalg.solve(\
                KW@self.K+lamb*np.eye(self.K.shape[0]),
                KW@z)

    # def _p_x(self):
    #     return 1/(1+np.exp(-self.K @ self.a)).reshape((-1,))

    def _z(self, y, W, p, Ka):
        return Ka + np.linalg.solve(W, y - p.reshape((-1,1)))

    def fit(self, x, y, lamb = 0.1, num_iters = 10):
        # Initialise
        if self.precomputed_kernel:
            assert x.shape[0] == x.shape[1], "precomputed_kernel requires a squared matrix"
        self.x = x.copy()
        self.a = np.zeros((x.shape[0], 1))
        self.K = self.kernel(x, x) if self.precomputed_kernel else self.x

        # Loop
        for _ in range(num_iters):
            a_old = self.a.copy()
            self._step(x, y, lamb)
            print(np.linalg.norm(self.a-a_old))
        
        # return self.a
    
    def decision_function(self, x_pred):
        """returns the raw scores before applying the logit function
        """
        ker = self.kernel(x_pred, self.x) if self.precomputed_kernel else x_pred
        return ker@self.a

    def predict_proba(self, x_pred):
        """returns the probability for the response y to be 1 (not the baseline)

        Args:
            x_pred (np.array): the explanatory x, can either be a kernel matrix in 
            with the training x (precomputed_kernel is True), or a raw feature vector (precomputed_kernel is False)
        """
        score = self.decision_function(x_pred)
        return 1/(1+np.exp(-score))#, ker@self.a > 0

    def predict(self, x_pred, prob_decision_boundary=0.5):
        probs = self.predict_proba(x_pred)
        y_pred = [1 if prob >= prob_decision_boundary else 0 for prob in probs]
        return np.array(y_pred)


