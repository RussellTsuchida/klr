import numpy as np

def invlogit(x):
    return 1/(1+np.exp(-x))
