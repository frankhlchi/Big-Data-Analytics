__author__ = 'chihongliang'
import numpy as np
#from PCA import asRowMatrix
from PCA import pca
import RPCA
from Project import project

# build eigenmodel (KNN+PCA) to take training set and find query image's nearest neighbor (based on PCA outcome)

class EigenModel(object):

    tx=None
    testx =None
    y=None
    projections=[]
    WW = []
    mu = []
    z=None


    def __init__(self, tx=None,y=None,z=None, num_components=50):
        [D, self.WW, self.mu] = pca(tx,num_components)
        self.y=y
        self.tx=tx
        self.z =z
        for xi in tx:
            self.projections.append(project(self.WW, xi.reshape(1,-1), self.mu))

#this method is for PART1 query since it only need to predict one class
    def predict(self, X):
        #in classification, use of RPCA will take a long time, therefore I put it in annotation
        #X= RPCA.robust_pca(X)
        testx =X
        minDist = np.finfo('float').max
        minClass = -1
        Q = project(self.WW, testx.reshape(1,-1),self.mu)
        rank = []
        k =[]
        for i in xrange(len(self.projections)):
            dist = self.EuclideanDistance(self.projections[i], Q)
            rank.append(dist)
            if dist < minDist:
                minDist = dist
                minClass = self.y[i]
                k.append(minClass)
        t =rank
        rank.sort();
        i =[]
        i.append(t.index(min(rank)))
        return minClass


#this method is for PART 2 since it needs to infer both class and superclass
    def predicttwo(self, X):
        #in classification, use of RPCA will take a long time, therefore I put it in annotation
        #X= RPCA.robust_pca(X)
        testx =X
        minDist = np.finfo('float').max
        minDistwo = np.finfo('float').max
        minClass = -1
        minClasstwo =-1
        Q = project(self.WW, testx.reshape(1,-1),self.mu)
        for i in xrange(len(self.projections)):
            dist = self.EuclideanDistance(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = self.y[i]

        for i in xrange(len(self.projections)):
            dist = self.EuclideanDistance(self.projections[i], Q)
            if dist < minDistwo:
                minDistwo = dist
                minClasstwo = self.z[i]

        return minClass,minClasstwo

    def EuclideanDistance(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p-q),2)))





