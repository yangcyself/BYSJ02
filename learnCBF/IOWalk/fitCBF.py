import numpy as np
from sklearn.svm import SVC
from glob import glob
import pickle as pkl
from scipy.optimize import minimize
import os
import json
import datetime
from ExperimentSecretary.Core import Session


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
class kernel:
    """
        a map to some higher dim space. E.g (x,y) -> (x^2,y^2,x,y)
    """
    @staticmethod
    def augment(x):
        """
            input is either 1dim dataponit, or a matrix where each row is a datapoint
            return the augmented matrix, where each row is the feature of a datapoint
        """
        x = np.array(x)
        if(x.ndim==1):
            x  = x.reshape((1,-1))
        x = x.T
        return np.concatenate([[a*b for i,a in enumerate(x) for b in x[i:] ],x,np.ones((1,x.shape[1]))],axis = 0).T

    @staticmethod
    def jac(x):
        """
            input is 1dim dataponit
            return the jacobian of this datapoint
        """
        x = np.array(x)
        return np.concatenate([ [ [(a if k==j+i else 0) + (b if k==i else 0)  for k in range(len(x))] 
                                for i,a in enumerate(x) for j,b in enumerate(x[i:]) ] ,np.eye(len(x)),np.zeros((1,len(x)))],axis=0)
    
    @staticmethod
    def GetParam(w,dim=20):
        """
            input the trained w
            Return the A, b, c of x^TAx + b^T x + c
        """
        print("w:",w)
        A = np.array([[w[min(i,j)*dim - int(min(i,j)*(min(i,j)-1)/2) - min(i,j) + max(i,j)]/(1 if(i==j)else 2)  for j in range(dim)] for i in range(dim)])
        b = w[-dim-1:-1]
        c = w[-1]
        return A,b,c



def fitCBF(X, y, X_c,dim = 20, x0 = None):
    """
        X: tested datapoints
        y: 1 or -1; 1 means in CBF's upper level set
        X_c: The datapoints that needs to ensure exits u s.t. dB(x,u) > 0

        cast it into an optimization problem, where
        min_{w,b,u}  ||w||+||u||
        s.t.    y_i(x_i^T w + b) > 1 //SVM condition
                y_i(dB(x_i,u)+ mc B(x_i)) > 0 for y_i > 0  //CBF defination
                u_min < u < u_max
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

    def completeCons(w):
        # the points at the boundary of the CBF needs to have a solution that dB > 0
        return np.array([1]+[w.T @ kernel.jac(x) @ Dyn_A @ x  +  
                    MAX_INPUT * np.linalg.norm(Dyn_B.T @ kernel.jac(x).T @ w, ord = 1) for x in X_c])

    options = {"maxiter" : 500, "disp"    : True}
    lenx0 = int((dim+1)*dim/2 + dim + 1)
    x0 = np.random.random(lenx0) if x0 is None else x0
    
    constraints = [{'type':'ineq','fun':SVMcons, "jac":SVMjac},
                   {'type':'ineq','fun':completeCons}]
    
    bounds = np.ones((lenx0,2)) * np.array([[-1,1]]) * 100
    bounds[-5:-1,:] *=0

    res = minimize(obj, x0, options = options,jac=grad, bounds=bounds,
                constraints=constraints, method =  'SLSQP') # 'trust-constr' , "SLSQP"

    # print("SVM Constraint:\n", SVMcons(res.x[:len(x0)]))
    return (*kernel.GetParam(res.x), res.x)


def fitCompleteCBF(X,y,dim = 20):
    """
        call the `fitCBF` iteratively, each step augment the dataset with some data points 
            that makes the optimization of u hard
    """
    x0 = None
    X_c = []
    for i in range(10):
        HA,Hb,Hc, x0 = fitCBF(X,y,X_c,dim,x0)
    
        def obj(x):
            DA,DB,Dg = getDynamic(x)
            return 2 * x.T @ HA @ (DA @ x + Dg) \
                    + Hb @ DA @ x + Hb @ Dg \
                    + GP.MAXTORQUE * np.abs(2 * DB.T @ HA @ x + DB.T @ Hb,ord = 1)

        def cons(x):
            return x.T @ HA @ x + Hb.T @ x + Hc

        constraints = {'type':'eq','fun': cons, "jac":jacobian(cons)}
        
        bounds = np.ones((4,2)) * np.array([[-1,1]]) * 3
        
        X_c = [ minimize(obj, np.random.random(4) * 3, bounds = bounds, #jac=jac, 
                        constraints=constraints, method =  'SLSQP').x for i in range(1) ]
        print([obj(x) for x in X_c])
        # print(X_c)
    return A,b,c




###############################
##### SESSION DEFINITION ######
###############################


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
        A,b,c = SVM_factors(np.array(self.X),self.y,gamma=self.gamma, dim=self.X.shape[1]-1, class_weight = self.class_weight)
        dumpJson(A,b,c,self.resultFile)
        test = [ x.T @ A @ x + b.T @ x + c for x in self.X_] 
        self.add_info("Test:TruePositive",float(np.sum([f*y >0 for f,y in zip(test,self.y) if y==1])/sum(np.array(self.y)==1)))
        self.add_info("Test:TrueNegative",float(np.sum([f*y >0 for f,y in zip(test,self.y) if y==-1])/sum(np.array(self.y)==-1)))

    

    @FitCBFSession_t.column
    def timeComsumption(self):
        return str(datetime.datetime.now() -  self._startTime)

if __name__ == '__main__':
    s = FitCBFSession("./data/StateSamples/IOWalkSample/2020-05-01-12_41_24",
        name = "weight_WithB",class_weight={1: 0.3, -1: 0.7})
    s()