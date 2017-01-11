__author__ = 'chihongliang'
import numpy as np
#this code is to rotate a picture and return it
def rotate(arr):
    matrix =np.empty((32,32,3))
    for i in range(32):
        for j in range(32):
            for n in range(3):
                matrix[i,j,n]=(arr[32*3*i+j*3+n])
    temp =np.empty((32,32,3))
    for i in range(32):
        for j in range(32):
            temp[j,i,:]=matrix[i,j,:]
    rotation =[0]*3072
    for i in range(32):
        for j in range(32):
            for n in range(3):
                rotation[32*3*i+j*3+n]=int(temp[i,j,n])
    return rotation


