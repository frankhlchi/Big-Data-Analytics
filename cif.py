__author__ = 'chihongliang'
import cPickle
import numpy as np

#this code is to load CIFAR-10 dataset
CIFAR_PATH = './cifar-10-batches-py'
CIFAR_FILES = ('data_batch_1',
               'data_batch_2',
               'data_batch_3',
               'data_batch_4',
               'data_batch_5')

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load():
    data = np.zeros((50000, 3072), 'uint8')
    for i in range(len(CIFAR_FILES)):
        dict = unpickle(CIFAR_PATH + '/'
                        + CIFAR_FILES[i])
        data[10000*i:10000*(i+1),:] = dict['data']

    return data

def loadlabel():
    data = np.zeros((50000))
    for i in range(len(CIFAR_FILES)):
        dict = unpickle(CIFAR_PATH + '/'
                        + CIFAR_FILES[i])
        data[10000*i:10000*(i+1)] = dict['labels']
    return data

def loadtest():
    data = np.zeros((10000, 3072), 'uint8')
    for i in range(len(CIFAR_FILES)):
        dict = unpickle(CIFAR_PATH + '/'
                        + 'test_batch')
        data[0:10000,:] = dict['data']
    return data

def loadtestlabel():
    data = np.zeros((10000), 'uint8')
    for i in range(len(CIFAR_FILES)):
        dict = unpickle(CIFAR_PATH + '/'
                        + 'test_batch')
        data[0:10000] = dict['labels']
    return data
