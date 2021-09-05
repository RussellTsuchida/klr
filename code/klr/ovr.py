from klr.klr import Klr
from klr.response_funcs import invlogit
from klr.helpers import SquaredExponential
from sklearn import preprocessing
import numpy as np
from scipy.sparse import csr_matrix

class Ovr():
    def __init__(self, x, y, lamb = 0.1, num_iters = 10, kernel_func=None, precomputed_kernel=True):
        """implement one versus rest klr classification

        Args:
            kernel_func ([type], optional): [description]. Defaults to None.
            precomputed_kernel (bool, optional): [description]. Defaults to True.
        """
        # Initialise
        self.x = x.copy()
        self.y = y.copy()
        self.lamb = lamb
        self.num_iters = num_iters
        if precomputed_kernel:
            assert x.shape[0] == x.shape[1], "precomputed_kernel requires a squared matrix"
            self.K = self.x
        else:
            self._init_kernel_func(kernel_func)
            self.K = self.kernel_func(x, x)

        # converting y into ovr
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(y) 
        self.ovrd_y = self.lb.transform(y) # each column of ovrd_y is the one-vs-rest response when 1 category is the nonbaseline (nrows=#obs, ncols=#categories)
            

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


    def fit(self):
        for i in range(self.ovrd_y.shape[1]):
            model = Klr(precomputed_kernel=True)
            model.fit(self.K,self.ovrd_y[:,i], lamb=self.lamb, num_iters=self.num_iters)
            yield model

    def decision_function(self, x_pred):
        scores = []
        for model in self.fit():
            score = model.decision_function(x_pred).flatten()
            scores.append(score)
        return np.array(scores).T # matrix with each column as the score vector when 1 category is the nonbaseline

    def predict_proba(self,x_pred):
        scores = self.decision_function(x_pred)
        return invlogit(scores)

    def predict(self, x_pred):
        probs = self.predict_proba(self,x_pred)
        best = np.argmax(probs, axis=1)
        num_obs = self.ovrd_y.shape[0]
        num_categories = self.ovrd_y.shape[1]
        y_pred = csr_matrix((np.repeat(1,num_obs),best,np.arange(0,num_obs,num_categories)), shape=(num_obs,num_categories))
        return self.lb.inverse_transform(y_pred)