import numpy as np
from sklearn.svm import SVC
from glob import glob
import pickle as pkl
from scipy.optimize import minimize
import os
import json
import datetime
from ExperimentSecretary.Core import Session

import sys
sys.path.append(".")
from learnCBF.FittingUtil import kernel
sqeuclidean = lambda x: np.inner(x, x) # L2 NORM SQUARE

import globalParameters as GP
GP.GUIVIS = False
GP.CATCH_KEYBOARD_INTERRPUT = False
from ctrl.CBFWalker import *
ct = CBF_WALKER_CTRL()
IOLinearCTRL.gc_mask.reg(ct,asSuper = True)
ct.restart()
    


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


def SVM_factors(X,y,dim = 20,gamma = 9999, class_weight = None):
    clf = SVC(kernel = "poly", degree=2, gamma=gamma,verbose=False, class_weight = class_weight)
    print("start Fitting")
    print("X shape:",X.shape)
    clf.fit(X, y)
    print("Finished Fitting")
    A,b,c = recoverSecondOrderFunc(lambda x : clf.decision_function(np.array( [1] + list(x)).reshape((1,-1)) ),dim = dim)
    # print("Function parameter \n",A,b,c)
    return A,b,float(c)


def dumpJson(A,b,c,fileName = "data/CBF/tmp2.json"):
    json.dump({"A":A.tolist(),"b":b.tolist(),"c":c},open(fileName,"w"))


def getDynamic(x):
    """
        return the dynamic matrix DA, DB, Dg given CBF feature state x
    """
    x = x[list(range(0,7))+list(range(10,17))]
    ct.setState(x)
    ct.resetFlags()
    return ct.DHA, ct.DHB, ct.DHg


###############################
########## FIT CBF ############
###############################

# Add more constraints to the fitting problem


def fitFeasibleCBF(X, y, u_list, mc = 1, dim = 20):
    """
        X: tested datapoints
        y: 1 or -1; 1 means in CBF's upper level set
        u: The input from the sample of each `X`, only the state with y=1 is used
        mc: gamma

        cast it into an optimization problem, where
        min_{w,b}  ||w||
        s.t.    y_i(x_i^T w + b) > 1 //SVM condition
                y_i(dB(x_i,u_i)+ mc B(x_i)) > 0 for y_i > 0  //CBF defination
    """
    
    print("START")
    def obj(w):
        return sqeuclidean(w[:-1])
    def grad(w):
        ans = 2*w
        ans[-1] = 0
        return ans.reshape((1,-1))

    X_aug = kernel.augment(X)
    y = np.array(y).reshape((-1))

    def SVMcons(w):
        return y * (X_aug @ w) - 1
    def SVMjac(w):
        # print(w)
        return y.reshape((-1,1))*X_aug

    # X_c_aug = kernel.augment(X_c) if len(X_c) else []
    feasiblePoints = [(x,xa,u,*getDynamic(x)) for x,xa,y,u in  zip(X,X_aug,y,u_list) if y ==1]
    
    # [w.T @ kernel.jac(x) @ Dyn_A @ x  +  
    #                 MAX_INPUT * np.linalg.norm(Dyn_B.T @ kernel.jac(x).T @ w, ord = 1) for x in X_c])
    def feasibleCons(w):
        return np.array([w.T @ kernel.jac(x) @ (A @ x + B @ u + g) + mc * xa @ w
                    for x,xa,u,A,B,g in feasiblePoints])
    # # TODO
    # def symmetricCons(w):
    #     pass

    options = {"maxiter" : 500, "disp"    : True}
    lenx0 = int((dim+1)*dim/2 + dim + 1)
    x0 = np.random.random(lenx0)

    constraints = [{'type':'ineq','fun':SVMcons, "jac":SVMjac},
                   {'type':'ineq','fun':feasibleCons}]
    
    bounds = np.ones((lenx0,2)) * np.array([[-1,1]]) * 100

    res = minimize(obj, x0, options = options,jac=grad, bounds=bounds,
                constraints=constraints, method =  'SLSQP') # 'trust-constr' , "SLSQP"

    # print("SVM Constraint:\n", SVMcons(res.x[:len(x0)]))
    return (*kernel.GetParam(res.x), res.x)



###############################
##### SESSION DEFINITION ######
###############################


class FitCBFSession_t(Session):
    """
        define the Session class to record the information
    """
    pass

class FitCBFSession(FitCBFSession_t):
    def __init__(self, loaddir, name = "tmp", gamma = 9999, class_weight = None, algorithm = "default"):
        super().__init__(expName="IOWalkFit")
        self.X = []
        self.y = []
        self.loadedFile = []
        self.gamma = gamma
        self.algorithm = algorithm.lower()
        self.class_weight = class_weight
        for f in glob(os.path.join(loaddir,"*.pkl"))[:40]:
            data = pkl.load(open(f,"rb"))
            self.loadedFile.append(f)
            self.X += list(data['safe'])
            self.y += [ 1 ] * len(data['safe'])
            self.X += list(data['danger'])
            self.y += [ -1 ] * len(data['danger'])

        self.X,self.u = [x for x,u in self.X], [u for x,u in self.X]

        self.X_ = np.array(self.X)[:,1:] # X_ is the 
        self.X = np.ones_like(self.X)
        self.X[:,1:] = self.X_ # set the first row of X to be 1

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
        if(self.algorithm == "default"):
            A,b,c = SVM_factors(np.array(self.X),self.y,gamma=self.gamma, dim=self.X.shape[1]-1, class_weight = self.class_weight)
        elif(self.algorithm == "feasible"):
            A,b,c,x0 =  fitFeasibleCBF(np.array(self.X),self.y,self.u,dim=self.X.shape[1]-1)
        dumpJson(A,b,c,self.resultFile)
        test = [ x.T @ A @ x + b.T @ x + c for x in self.X_] 
        self.add_info("Test:TruePositive",float(np.sum([f*y >0 for f,y in zip(test,self.y) if y==1])/sum(np.array(self.y)==1)))
        self.add_info("Test:TrueNegative",float(np.sum([f*y >0 for f,y in zip(test,self.y) if y==-1])/sum(np.array(self.y)==-1)))

    
    @FitCBFSession_t.column
    def timeComsumption(self):
        return str(datetime.datetime.now() -  self._startTime)

if __name__ == '__main__':
    s = FitCBFSession("./data/StateSamples/IOWalkSample/2020-05-07-23_33_30",
        name = "Feasible",algorithm="feasible")
    s()