from pyflann import FLANN
import numpy as np

def Euclidean(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.linalg.norm(train[x]-test[y])
    return D

feature_difference = Euclidean

def NV_Euclidean(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            distance = train[x] - test[y,2]
            value = np.linalg.norm(distance, axis=1, keepdims=True)
            D[x,y] = np.min(value)
    return D

def N2One_Euclidean(train, test):
    print (train.shape)
    D = np.zeros([train.shape[0], test.shape[0]])
    opt_feature = np.zeros_like(test)
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            distance = train[x] - test[y]
            value = np.linalg.norm(distance, axis=1, keepdims=True)
            opt_feature[x] = train[x, np.argmin(value)]
            D[x,y] = np.min(value)
    return D

def Manhattan(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.sum(abs(train[x]-test[y]))
    return D

def Chebyshev(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.max(abs(train[x]-test[y]))
    return D

def Cosine(train, test):
    D = np.zeros([train.shape[0], test.shape[0]])
    for x in range(train.shape[0]):
        for y in range(test.shape[0]):
            D[x,y] = np.sum(train[x]*test[y])/(np.linalg.norm(train[x])*np.linalg.norm(test[y]))
    return D

def getANN(data, test, k=30):
    flann = FLANN()
    result, dists = flann.nn(np.array(data), np.array(test), \
                             k, algorithm="kmeans", branching=32, iterations=10, checks=16)
    return result, dists

