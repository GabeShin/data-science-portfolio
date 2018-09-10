import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle


def y2indicator(y):
    """
    Convert to indicator variable
    example:
    {1, 3, 2} -> {{0,1,0,0}, {0,0,0,1}, {0,0,1,0}}
    """
    N = len(y)
    ind = np.zeros((N,10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p, t):
    """
    Return error rate
    Compare prediction(p) to target(t)
    """
    return np.mean(p != t)

def flatten(X):
    """
    Flatten the input of MAT file.
    """
    N = X.shape[-1]
    flat = np.zeros((N, 3072))
    for i in range(N):
        flat[i] = X[:,:,:,i].reshape(3072)
    return flat

def getTrainData():
    train = loadmat('../input/SVHN Dataset/train_32x32.mat')

    Xtrain = flatten(train['X'].astype(np.float32) / 255.)
    Ytrain = train['y'].flatten() - 1
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    return Xtrain, Ytrain

def getTestData():
    test = loadmat('../input/SVHN Dataset/test_32x32.mat')

    Xtest  = flatten(test['X'].astype(np.float32) / 255.)
    Ytest  = test['y'].flatten() - 1
    
    return Xtest, Ytest
