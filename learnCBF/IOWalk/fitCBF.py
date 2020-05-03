import numpy as np
from sklearn.svm import SVC
from glob import glob
import pickle as pkl
from scipy.optimize import minimize
import os
import json


def recoverSecondOrderFunc(f,dim = 20):
    # retrive x^T A x + b^T x + c = decision_func
    c = f(np.zeros(dim))
    b = np.zeros(dim)
    A = np.zeros((dim,dim))
    for i in range(dim):
        t1,t2 = np.zeros(dim),np.zeros(dim)
        t1[i] = 1
        t2[i] = -1
        b[i] = (f(t1)-f(t2))/2
        A[i,i] = (f(t1)+f(t2))/2 - c
    for i in range(dim):
        for j in range(i):
            t = np.zeros(dim)
            t[i] = t[j] = 1
            A[i,j] = A[j,i] = (f(t) - t.T @ A @ t - b @ t - c)/2
    return A,b,c

def SVM_factors(X,y,dim = 20):
    clf = SVC(kernel = "poly", degree=2, gamma=99999,verbose=True,cache_size=7000 )
    print("start Fitting")
    clf.fit(X, y)
    print("Finished Fitting")
    A,b,c = recoverSecondOrderFunc(lambda x : clf.decision_function(x.reshape((1,-1))),dim = dim)
    # print("Function parameter \n",A,b,c)
    return A,b,float(c)

def dumpJson(A,b,c,fileName = "data/CBF/tmp2.json"):
    json.dump({"A":A.tolist(),"b":b.tolist(),"c":c},open(fileName,"w"))


if __name__ == '__main__':

    loaddir = "./data/StateSamples/IOWalkSample/2020-05-01-12_41_24"
    X = []
    y = []
    for f in glob(os.path.join(loaddir,"*.pkl"))[:45]:
        data = pkl.load(open(f,"rb"))
        X += list(data['safe'])
        y += [ 1 ] * len(data['safe'])
        X += list(data['danger'])
        y += [ -1 ] * len(data['danger'])
    print(np.array(X).shape)
    A,b,c = SVM_factors(np.array(X),y)
    dumpJson(A,b,c)