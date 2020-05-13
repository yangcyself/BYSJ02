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


def fitFeasibleCBF(X, y, u_list, mc, dim, gamma, gamma2, class_weight= None):
    """
        X: tested datapoints
        y: 1 or -1; 1 means in CBF's upper level set
        u: The input from the sample of each `X`, only the state with y=1 is used
        mc: the constraint of dB + mc B > 0
        gamma: The gamma in SVM objective

        cast it into an optimization problem, where
        min_{w,b}  ||w||
        s.t.    y_i(x_i^T w + b) > 1 //SVM condition
                y_i(dB(x_i,u_i)+ mc B(x_i)) > 0 for y_i > 0  //CBF defination
    """
    
    print("START")
    class_weight = {-1:1, 1:1} if class_weight is None else class_weight

    def obj(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        # print("gamma * np.sum(np.clip(c,0,None))",gamma * np.sum(np.clip(c,0,None)))
        return sqeuclidean(w[:-1]) + gamma * np.sum(c) + gamma2 * sqeuclidean(u_ - uvec)
    def grad(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        ans = 2*w
        ans[-1] = 0
        # print("c,gamma,", c,gamma)
        # print("list(gamma*(c>0))",list(gamma*(c>0)))
        ans = np.array(list(ans)+list([gamma]*len(c)) +list(2*gamma2*(u_ - uvec)))
        return ans.reshape((1,-1)) 

    X_aug = kernel.augment(X)
    y = np.array(y).reshape((-1))
    lenw = int((dim+1)*dim/2 + dim + 1)
    print("X_aug shape", X_aug.shape)
    print("y shape", y.shape)
    def SVMcons(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return y * (X_aug @ w) - 1 + c
    def SVMjac(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return np.concatenate([y.reshape((-1,1))*X_aug, np.eye(len(c)),np.zeros((len(c),lenfu)) ],axis=1)


    # X_c_aug = kernel.augment(X_c) if len(X_c) else []
    feasiblePoints = [(x,xa,u,*getDynamic(x)) for x,xa,y,u in  zip(X,X_aug,y,u_list) if y ==1]
    udim = len(u_list[0])
    print("udim:",udim)
    lenfu = len(feasiblePoints) * udim # The length of all the `du`
    uvec = np.concatenate([u for (x,xa,u,A,B,g) in feasiblePoints],axis = 0) # make the array of u a vector
    assert(lenfu == len(uvec))
    print("len(feasiblePoints)",len(feasiblePoints))
    # [w.T @ kernel.jac(x) @ Dyn_A @ x  +  
    #                 MAX_INPUT * np.linalg.norm(Dyn_B.T @ kernel.jac(x).T @ w, ord = 1) for x in X_c])

    def feasibleCons(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return np.array([w.T @ kernel.jac(x) @ (A @ x + B @ (u_[i*udim:(i+1)*udim]) + g) + mc * xa @ w
                    for i,(x,xa,u,A,B,g) in enumerate(feasiblePoints)])

    def feasibleJac(w):        
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return np.concatenate([ 
                np.array(list(kernel.jac(x) @ (A @ x + B @ (u_[i*udim:(i+1)*udim]) + g) + mc * xa)
                    +[0]*(len(c)) + [0]*(i*udim)+ list(w.T@ kernel.jac(x)@B) +[0]*(lenfu-(i*udim)-udim)
                ).reshape((1,-1))
                for i,(x,xa,u,A,B,g) in enumerate(feasiblePoints)],axis=0)

    # # TODO
    # def symmetricCons(w):
    #     return

    options = {"maxiter" : 500, "disp"    : True, 'ftol': 1e-06,'iprint':2, "verbose":1}
    lenx0 = lenw + len(y) + lenfu
    x0 = np.random.random(lenx0)
    x0[lenw:-lenfu]  *= 0 # set the init of c to be zero
    x0[-lenfu:] = uvec # set the init of u_ to be u

    constraints = [{'type':'ineq','fun':SVMcons, "jac":SVMjac}]#,
                #    {'type':'ineq','fun':feasibleCons, "jac":feasibleJac}]
    
    bounds = np.ones((lenx0,2)) * np.array([[-1,1]]) * 9999
    bounds[0,:] *= 0 # the first dim `x` 
    bounds[lenw:-lenfu,0] = 0 # set c > 0
    bounds[lenw:-lenfu,1] = np.inf
    bounds[-lenfu:,0] = -np.array(list(GP.MAXTORQUE)*len(feasiblePoints))
    bounds[-lenfu:,1] =  np.array(list(GP.MAXTORQUE)*len(feasiblePoints))
    res = minimize(obj, x0, options = options,jac=grad, bounds=bounds,
                constraints=constraints, )#method =  'SLSQP') # 'trust-constr' , "SLSQP"

    # print("SVM Constraint:\n", SVMcons(res.x[:len(x0)]))
    return (*kernel.GetParam(res.x), res)



###############################
##### SESSION DEFINITION ######
###############################


class FitCBFSession_t(Session):
    """
        define the Session class to record the information
    """
    pass

class FitCBFSession(FitCBFSession_t):
    def __init__(self, loaddir, name = "tmp", mc = 0.01,  gamma = 1, gamma2=100, class_weight = None, algorithm = "default", trainNum = 10):
        super().__init__(expName="IOWalkFit")
        self.X = []
        self.y = []
        self.loadedFile = []
        self.mc = mc
        self.gamma = gamma
        self.gamma2 = gamma2
        self.algorithm = algorithm.lower()
        self.class_weight = class_weight
        self.trainNum = trainNum
        for f in glob(os.path.join(loaddir,"*.pkl")):
            data = pkl.load(open(f,"rb"))
            self.loadedFile.append(f)
            self.X += list(data['safe'])
            self.y += [ 1 ] * len(data['safe'])
            self.X += list(data['danger'])
            self.y += [ -1 ] * len(data['danger'])
        
        self.X_train, self.y_train, self.u_train = np.array([x for x,u in self.X[:self.trainNum]]), self.y[:self.trainNum], np.array([u for x,u in self.X[:self.trainNum]])
        self.X_test, self.y_test, self.u_test = np.array([x for x,u in self.X[self.trainNum:]]), self.y[self.trainNum:],  np.array([u for x,u in self.X[self.trainNum:]])
        self.X = np.array([x for x,u in self.X])

        self.resultFile = "data/CBF/%s_%s.json"%(name,self._storFileName)

        self.add_info("resultFile",self.resultFile)
        self.add_info("Loaded File",self.loadedFile)
        self.add_info("Total Points",len(self.X))
        self.add_info("Total Training Points",len(self.X_train))
        self.add_info("X dim",self.X.shape[1])
        self.add_info("number Positive",int(sum(np.array(self.y)==1)))
        self.add_info("mc",self.mc)
        self.add_info("gamma",self.gamma)
        self.add_info("gamma2",self.gamma2)
        self.add_info("class_weight",self.class_weight)
        self.add_info("trainNum",self.trainNum)
        self.add_info("class_weight",self.class_weight)
        self.add_info("trainNum",self.trainNum)
    

    def body(self):
        NoKeyboardInterrupt = True
        try:
            self._startTime = datetime.datetime.now()
            if(self.algorithm == "default"):
                X_ = np.array(self.X_train)[:,1:] # X_ is the 
                X = np.ones_like(X)
                X[:,1:] = X_ # set the first row of X to be 1, otherwise the scipy SVM cannot fit the vector b
                A_,b_,c = SVM_factors(np.array(X),self.y_train,gamma=self.gamma, dim=self.X_train.shape[1]-1, class_weight = self.class_weight)
                A = np.zeros((A_.shape[0]+1, A_.shape[1]+1))
                A[1:,1:] = A_
                b = np.zeros(len(b_)+1)
                b[1:] = b_
                Traintest = [ x.T @ A @ x + b.T @ x + c for x in self.X_train] 
                Testtest = [x.T @ A @ x + b.T @ x + c for x in self.X_test]

            elif(self.algorithm == "feasible"):
                A,b,c,res =  fitFeasibleCBF(np.array(self.X_train),self.y_train,self.u_train,dim=self.X_train.shape[1], 
                                    mc = self.mc, gamma=self.gamma, gamma2=self.gamma2, class_weight = self.class_weight)
                self.add_info("Optimization_TerminationMessage",res.message)
                Traintest = [ x.T @ A @ x + b.T @ x + c for x in self.X_train]
                Testtest = [ x.T @ A @ x + b.T @ x + c for x in self.X_test]

            dumpJson(A,b,c,self.resultFile)
            self.add_info("TrainingTest:TruePositive",float(np.sum([f*y >0 for f,y in zip(Traintest,self.y_train) if y==1])/sum(np.array(self.y_train)==1)))
            self.add_info("TrainingTest:TrueNegative",float(np.sum([f*y >0 for f,y in zip(Traintest,self.y_train) if y==-1])/sum(np.array(self.y_train)==-1)))
            self.add_info("TestingTest:TruePositive",float(np.sum([f*y >0 for f,y in zip(Testtest,self.y_test) if y==1])/sum(np.array(self.y_test)==1)))
            self.add_info("TestingTest:TrueNegative",float(np.sum([f*y >0 for f,y in zip(Testtest,self.y_test) if y==-1])/sum(np.array(self.y_test)==-1)))
        
        except KeyboardInterrupt as ex:
            NoKeyboardInterrupt = False
        assert NoKeyboardInterrupt, "KeyboardInterrupt caught"    
    
    @FitCBFSession_t.column
    def timeComsumption(self):
        return str(datetime.datetime.now() -  self._startTime)

if __name__ == '__main__':
    s = FitCBFSession("./data/StateSamples/IOWalkSample/2020-05-10-09_03_08",
        name = "tmp",algorithm="feasible", trainNum=80)
    s()