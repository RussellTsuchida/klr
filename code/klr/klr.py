import numpy as np
from klr.helpers import SquaredExponential, identity
from klr.response_funcs import invlogit
from functools import partial

class Klr(object):
    """klr implementation based on 
    https://papers.nips.cc/paper/2001/file/2eace51d8f796d04991c831a07059758-Paper.pdf,
    with similar annotation of variables

    Args:
        object ([type]): [description]
    """
    def __init__(self, kernel_func=None, precomputed_kernel=True, use_solver='numpy'):
        self._detect_solver(use_solver)
        self.K = None
        self.x = np.asarray([np.nan])
        self.precomputed_kernel = precomputed_kernel
        if not self.precomputed_kernel:
            self._init_kernel_func(kernel_func)

    def _detect_solver(self, use_solver):
        if use_solver == 'pyarma':
            import pyarma
            self.solver = pyarma.solve
            self._mat_convert = pyarma.mat
        elif use_solver == 'numpy':
            self.solver = np.linalg.solve
            self._mat_convert = identity
        elif use_solver == 'scipy':
            import scipy.linalg
            # it should be possible to use 'pos' instead of 'her', 
            # but sometimes the matrices are numerically singular and 
            # 'pos' fails
            self.solver = partial(scipy.linalg.solve, assume_a='her')
            self._mat_convert = identity

    def _init_kernel_func(self, kernel_func):
        """specify the function to compute kernels

        Args:
            kernel (func ?): kernel function
        """
        if kernel_func is None:
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
        self.a = self.solver(self._mat_convert(KW@self.K+lamb*self.K),
                self._mat_convert(KW@z))

    def _z(self, y, W, p, Ka):
        return Ka + self.solver(self._mat_convert(W), 
            self._mat_convert(y - p.reshape((-1,1))))

    def fit(self, x, y, lamb = 0.1, num_iters = 10):
        # Initialise
        if self.precomputed_kernel:
            assert x.shape[0] == x.shape[1], "precomputed_kernel requires a squared matrix"

        self.a = np.zeros((x.shape[0], 1))
        # Sometimes we fit multiple times using the same kernel. In this case,
        # don't keep computing the kernel every time.
        if (self.K is None) or not (np.array_equal(x, self.x)):
            self.x = x.copy()
            self.K = self.kernel_func(x, x) if not self.precomputed_kernel else self.x

        # Loop
        for _ in range(num_iters):
            a_old = self.a.copy()
            self._step(y, lamb)
            #print(np.linalg.norm(self.a-a_old))
        
    def decision_function(self, x_pred):
        """returns the raw scores before applying the logit function
        """
        ker = self.kernel_func(x_pred, self.x) if not self.precomputed_kernel else x_pred
        return ker@self.a

    def predict_proba(self, x_pred):
        """returns the probability for the response y to be 1 (not the baseline)

        Args:
            x_pred (np.array): the explanatory x, can either be a kernel matrix in 
            with the training x (precomputed_kernel is True), or a raw feature vector (precomputed_kernel is False)
        """
        score = self.decision_function(x_pred)
        return invlogit(score)

    def predict(self, x_pred, prob_decision_boundary=0.5):
        probs = self.predict_proba(x_pred).flatten()
        y_pred = probs >= prob_decision_boundary
        return np.array(y_pred).reshape(-1,1)


