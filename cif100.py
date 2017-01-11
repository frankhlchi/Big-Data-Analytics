__author__ = 'chihongliang'
import cPickle

#this code is to load CIFAR_100 dataset
CIFAR_PATH = './cifar-100-python'
CIFAR_FILES = ('train',
               'test',
               'meta',)

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def traindata():
    data = unpickle(CIFAR_PATH + '/'+ CIFAR_FILES[0])['data']
    return data

def trainfinelabel():
    data = unpickle(CIFAR_PATH + '/'+ CIFAR_FILES[0])['fine_labels']
    return data

def traincoarselabel():
    data = unpickle(CIFAR_PATH + '/'+ CIFAR_FILES[0])['coarse_labels']
    return data

def testdata():
    data = unpickle(CIFAR_PATH + '/'+ CIFAR_FILES[1])['data']
    return data

def testfinelabel():
    data = unpickle(CIFAR_PATH + '/'+ CIFAR_FILES[1])['fine_labels']
    return data

def testcoarselabel():
    data = unpickle(CIFAR_PATH + '/'+ CIFAR_FILES[1])['coarse_labels']
    return data

def meta():
    data = unpickle(CIFAR_PATH + '/'+ CIFAR_FILES[2])
    return data
