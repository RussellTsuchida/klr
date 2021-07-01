import numpy as np
from ..klr import Klr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Synthetic data
    X1 = np.random.normal(-1, 1, (100,2))
    X2 = np.random.normal(1, 1, (201,2))
    Y1 = np.zeros((X1.shape[0],1))
    Y2 = np.ones((X2.shape[0],1))

    X = np.vstack((X1, X2))
    Y = np.vstack((Y1, Y2))

    plt.scatter(X1[:,0], X1[:,1], c='b', s=1)
    plt.scatter(X2[:,0], X2[:,1], c='r', s=1)

    # Model and model predictions
    model = Klr(None)
    model.fit(X, Y)
    x1 = np.linspace(np.amin(X[:,0]), np.amax(X[:,0]), 100)
    x2 = np.linspace(np.amin(X[:,1]), np.amax(X[:,1]), 100)

    xstar = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))]) 
    c, labels = model.predict(xstar)

    c[np.where(np.isclose(c,0.5, atol=0.05))]=np.nan
    plt.scatter(xstar[:,0], xstar[:,1], c=c, zorder=-5, alpha=0.2)

    
    # Accuracy on training data
    _, Y_predict = model.predict(X)
    print(np.sum(Y_predict == Y)/Y.shape[0])
    plt.savefig('klr.pdf')
