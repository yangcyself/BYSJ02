import numpy as np
from sklearn.svm import SVC
from glob import glob
import pickle as pkl
from scipy.optimize import minimize
import os
import json
import datetime
from ExperimentSecretary.Core import Session


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

def SVM_factors(X,y,dim = 20,gamma = 9999):
    clf = SVC(kernel = "poly", degree=2, gamma=gamma,verbose=False)
    print("start Fitting")
    print("X shape:",X.shape)
    clf.fit(X, y)
    print("Finished Fitting")
    A,b,c = recoverSecondOrderFunc(lambda x : clf.decision_function(x.reshape((1,-1))),dim = dim)
    # print("Function parameter \n",A,b,c)
    return A,b,float(c)

def dumpJson(A,b,c,fileName = "data/CBF/tmp2.json"):
    json.dump({"A":A.tolist(),"b":b.tolist(),"c":c},open(fileName,"w"))

class FitCBFSession_t(Session):
    """
        define the Session class to record the information
    """
    pass

class FitCBFSession(FitCBFSession_t):
    def __init__(self, loaddir, name = "tmp", gamma = 9999, class_weight = None):
        super().__init__(expName="IOWalkFit")
        self.X = []
        self.y = []
        self.loadedFile = []
        self.gamma = gamma
        self.class_weight = class_weight
        for f in glob(os.path.join(loaddir,"*.pkl"))[:40]:
            data = pkl.load(open(f,"rb"))
            self.loadedFile.append(f)
            self.X += list(data['safe'])
            self.y += [ 1 ] * len(data['safe'])
            self.X += list(data['danger'])
            self.y += [ -1 ] * len(data['danger'])

        self.X = np.array(self.X)[:,1:]

        self.resultFile = "data/CBF/%s_%s.json"%(name,self._storFileName)

        self.add_info("resultFile",self.resultFile)
        self.add_info("Loaded File",self.loadedFile)
        self.add_info("Total Points",len(self.X))
        self.add_info("X dim",self.X.shape[1])
        self.add_info("number Positive",int(sum(np.array(self.y)==1)))
        self.add_info("gamma",self.gamma)
        self.add_info("class_weight",self.class_weight)
    

    def body(self):
        self._startTime = datetime.datetime.now()
        A,b,c = SVM_factors(np.array(self.X),self.y,gamma=self.gamma, dim=self.X.shape[1])
        dumpJson(A,b,c,self.resultFile)
    

    @FitCBFSession_t.column
    def timeComsumption(self):
        return str(datetime.datetime.now() -  self._startTime)

if __name__ == '__main__':
    s = FitCBFSession("./data/StateSamples/IOWalkSample/2020-05-01-12_41_24",
        name = "3_7weight",class_weight={1: 0.3, -1: 0.7})
    s()