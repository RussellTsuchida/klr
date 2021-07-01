import numpy as np
from abc import ABC, abstractmethod

class Distance(ABC):
    @abstractmethod
    def __call__(self, x1, x2):
        pass

    def _diff_sq(self, x1, x2):
        return ((x1[:, :, None] - x2[:, :, None].T)**2).sum(1)

class Euclidean(Distance):
    def __init__(self, squared_bool):
        self.squared_bool = squared_bool

    def __call__(self, x1, x2):
        diff_sq = self._diff_sq(x1, x2)
        if self.squared_bool:
            return diff_sq
        else:
            return np.sqrt(diff_sq)
        

class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1, x2):
        pass

    @staticmethod
    def _set_x2(x1, x2):
        if x2 is None:
            x2 = x1
        return x2

class SquaredExponential(Kernel):
    def __init__(self, lengthscale):
        self.lengthscale = lengthscale
        self.dist_sq = Euclidean(squared_bool = True)

    def __call__(self, x1, x2=None):
        x2 = self._set_x2(x1, x2)
        return np.exp(-self.dist_sq(x1, x2)/(2*self.lengthscale**2))


