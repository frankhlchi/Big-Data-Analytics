__author__ = 'chihongliang'
import numpy as np
import math
#the implementation of robust principle component analysis
def robust_pca(M):
    L = np.zeros(M.shape)
    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)
    print M.shape
    mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    while not converged(M,L,S):
        L = svd_shrink(M - S - (mu**-1) * Y, mu)
        S = shrink(M - L + (mu**-1) * Y, lamb * mu)
        Y = Y + mu * (M - L - S)
    return L

def svd_shrink(X, tau):
    U,s,V = np.linalg.svd(X, full_matrices=False)
    return np.dot(U, np.dot(np.diag(shrink(s, tau)), V))

def shrink(X, tau):
    V = np.copy(X).reshape(X.size)
    for i in xrange(V.size):
        V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
        if V[i] == -0:
            V[i] = 0
    return V.reshape(X.shape)

def frobeniusNorm(X):
    accum = 0
    V = np.reshape(X,X.size)
    for i in xrange(V.size):
        accum += abs(V[i] ** 2)
    return math.sqrt(accum)

def L1Norm(X):
    return max(np.sum(X,axis=0))

def converged(M,L,S):
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    print "error =", error
    return error <= 5e-5