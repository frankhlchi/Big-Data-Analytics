__author__ = 'chihongliang'

'''
this code is used to generate confusion matrix
'''

import numpy as np
from scipy.sparse import coo_matrix

def confusion_matrix(y_true, y_pred, labels):
    labels = np.asarray(labels)
    n_labels = labels.size
    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]

    CM = coo_matrix((np.ones(y_true.shape[0], dtype=np.int), (y_true, y_pred)),
                    shape=(n_labels, n_labels)
                    ).toarray()

    return CM
