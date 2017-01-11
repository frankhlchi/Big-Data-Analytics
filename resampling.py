__author__ = 'chihongliang'
import cif
import transform
import csv
import numpy as np
import RobustPCA

'''
we generate 2000 rotated pics for class 4 and 6 since they are easy to misclassify
'''
[X,y] = [cif.load(),cif.loadlabel()]
count =0
index=[]
for label in y:
    if label == 3:
        index.append(count)
    count =count +1
counttwo =0
indextwo=[]
for label in y:
    if label == 5:
        indextwo.append(counttwo)
    counttwo =counttwo +1
data = []
la =[]
for i in range(1000):
    row =transform.rotate(X[index[i]])
    data.append(row)
    la.append(3)
for i in range(1000):
    row =transform.rotate(X[indextwo[i]])
    data.append(row)
    la.append(5)
combineddata = np.zeros((52000, 3072), 'uint8')
combineddata[:50000,:]= cif.load()
combineddata[50000:,:]=data
combinedlabel= np.zeros((52000))
combinedlabel[:50000]=cif.loadlabel()
combinedlabel[50000:52000]=la

def expandeddata():
    return combineddata

def expandedlabel():
    return combinedlabel