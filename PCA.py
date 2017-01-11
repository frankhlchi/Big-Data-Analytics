__author__ = 'chihongliang'
import numpy as np
from scipy.linalg import svd

# PCA implementation
#ouput eigenvalues eigenvectors and mean vector
def pca(X, y=50, rand=np.random.RandomState(123)):
    r, q = X.shape
    X = X.astype(np.float64, copy=False)
    eps = 1e-6

    if r == 1:
        V = np.zeros((q, y), dtype=np.float64)
        M = X.flatten()
        return np.zeros((1, y), dtype=np.float64), V, M
    else:
        M = np.mean(X, axis=0)
        X -= M
        i = 2500
        if min(q, r) > i:
            X = X[rand.permutation(r)[:i]]
            r = i
        if q > r:
            V, S, _ = rsvd(np.dot(X, X.T) / (r - 1), rand=rand)
            s = [1.0 / np.sqrt(item) if abs(item) > eps else 0.0 for item in S]
            V = np.dot(np.dot(X.T, V), np.diag(s)) / np.sqrt(r - 1)
        else:
            V, S, _ = svd(np.dot(X.T, X) / (r - 1), rand=rand)
        V = V[:, S > eps]
        V = V[:, :y]
        P = np.zeros((q, y), dtype=np.float64)
        P[:, :V.shape[1]] = V
        return np.dot(X, P), P, M


#implementation of singular value decompositon
def rsvd(X, t=100, rand=50):
    try:
        U, S, V = svd(X, full_matrices=False)
    except Exception as e:
        if t <= 0:
            raise e
        else:
            len = X.size
            index = rand.random_integers(low=0, high=len-1)
            X[index / X.shape[1], index % X.shape[1]] += np.spacing(1)
            U, S, V =rsvd(X, t - 1, rand)
    return U, S, V

