import sys
sys.path.append(".")
import numpy as np
import globalParameters as GP
GP.GUIVIS = False
GP.CATCH_KEYBOARD_INTERRPUT = False
from scipy.linalg import sqrtm,expm
from ctrl.CBFWalker import *
from learnCBF.IOWalk.IOWalkUtil import *
from learnCBF.FittingUtil import kernel, sqeuclidean, dumpJson
import dill as pkl
from ExperimentSecretary.Core import Session
from glob import glob
import json
import datetime

import multiprocessing
from multiprocessing import Pool


def representEclips(A,b,c):
    """
    represent an Eclips using the end of its axises
    """
    sign = -1 # assume A is neg definate
    A,b,c = sign * A, sign * b, sign * c
    A += np.eye(A.shape[0])
    A_inv = np.linalg.inv(A)
    c = c - 1/4 * b.T @ A_inv @ b
    b = - 1/2 * A_inv @ b
    R = sqrtm(A_inv)
    E,V = np.linalg.eig(R)
    points = [b + np.sqrt(max(0,-c)) * e * vv for e,v in zip(E,V.T) for vv in [v,-v]] 
    return points,E,-c


def sampler(CBF, BalphaStd, Balpha = Balpha, CBFList = None):
    """
    sample a lot of trajectories, stop when the CBF constraint cannot be satisfied
        return sampled trajectories
    """
    HA,Hb,Hc = CBF

    ct = CBF_WALKER_CTRL()
    reset(ct)
    Balpha = Balpha + BalphaStd*(0.5-np.random.random(size = Balpha.shape))
    Kp = 200 + 400*np.random.rand()
    Kd = np.log(Kp)*np.random.rand()*4
    CBF_WALKER_CTRL.IOcmd.reg(ct,Balpha = Balpha)
    CBF_WALKER_CTRL.IOLin.reg(ct,kp = Kp, kd = Kd)
    # CBFList = [] if CBFList is None else CBFList
    # [ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF) for HA_CBF,Hb_CBF,Hc_CBF in CBFList]
    ct.addCBF(HA,Hb,Hc)

    Traj = []
    def callback_traj(obj):
        Traj.append((obj.t, obj.state, obj.Hx, obj.CBF_CLF_QP, (obj.DHA, obj.DHB, obj.DHg)))
    ct.callbacks.append(callback_traj)
    def callback_break(obj):
        return not (obj.LOG_CBF_SUCCESS and obj.LOG_CBF_ConsValue[0]>0)
    ct.callbacks.append(callback_break)
    ct.step(5)
    return Traj


def GetPoints(traj,CBF, dangerDT, safeDT):
    """
    get safe points and danger points given a trajectory.
        The safe points are the points before `safeDT` of the termination
        The danger points are the points got by re
    """
    HA,Hb,Hc = CBF

    # the danger points
    DcA,DcB,Dcg = traj[-1][4] # the Continues dynamics of the terminal point
    tt,ts,tx,tu,_ = traj[-1] # the time, state, augmented state, inputu of the terminal state
    if(tt<1.6*safeDT):
        return [],[]
    # Note, the `-DT` means that the time goes backward
    Dcg = Dcg.reshape(-1,1)
    sysdt = expm(np.concatenate([ - dangerDT * np.concatenate([DcA, DcB, Dcg],axis = 1),np.zeros((DcB.shape[1]+1, DcA.shape[1] + DcB.shape[1] + 1))], axis = 0))
    DA,DB,Dg = sysdt[:DcA.shape[0],:DcA.shape[1]], sysdt[:DcA.shape[0],DcA.shape[1]:-1], sysdt[:DcA.shape[0],-1:] # The dynamics with the time goes backwards
    sysdtf = expm(np.concatenate([ dangerDT * np.concatenate([DcA, DcB, Dcg],axis = 1),np.zeros((DcB.shape[1]+1, DcA.shape[1] + DcB.shape[1] + 1))], axis = 0))
    DAf,DBf,Dgf = sysdtf[:DcA.shape[0],:DcA.shape[1]], sysdtf[:DcA.shape[0],DcA.shape[1]:-1], sysdtf[:DcA.shape[0],-1:] # The dynamics with the time goes forwards
    Dg = Dg.reshape(-1)
    # find the `u` that maximizes the DBF (in continues dynamics)
    u = GP.MAXTORQUE * np.sign(((2*tx .T @ HA + Hb.T)@DBf).T)
    x_danger = [(DA @ tx + DB @ uu + Dg,uu,(DAf,DBf,Dgf))  for uu in [u, 2*u]] # 2*u means goes back with larger torque also means cannot be saved by u

    # The safe Points
    x_safe = [(traj[i][2],traj[i][3],traj[i][4]) for i in [int(-safeDT/GP.DT),int(-1.5*safeDT/GP.DT)]]
    return x_danger,x_safe


def getAsample(CBF,dangerDT,safeDT):
    return GetPoints(sampler(CBF,BalphaStd = 0.03),CBF,dangerDT,safeDT)


def learnCBFIter(CBF, badpoints, mc, dim, gamma0, gamma, gamma2, numSample, dangerDT, safeDT, class_weight= None, pool = None):
    """
    One iteration of the learnCBF: input a CBF, calls sampler and GetPints, and return a CBF
    
    Solve a optimization problem:
        feasible SVM with constraint: New CBF should be contained in old CBF

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
    HA,Hb,Hc = CBF

    # try: DEBUGGING CLOUSE
    #     samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    #     print("loaded %s from %s"%("samples", "./data/learncbf/tmp.pkl"))
    # except FileNotFoundError as ex:

    if pool is None:
        samples = [GetPoints(sampler(CBF,BalphaStd = 0.03),CBF,dangerDT,safeDT) for i in range(numSample)]
    else:
        
        samples = pool.map(getAsample, [(CBF,dangerDT,safeDT)]*numSample)
    #     pkl.dump(samples,open("./data/learncbf/tmp.pkl","wb"))

    X = [x for danger_s, safe_s in samples for x,u,DB in danger_s+safe_s]
    y = [i for danger_s, safe_s in samples for i in ([-1]*len(danger_s)+[1]*len(safe_s))]
    u_list = [u for danger_s, safe_s in samples for x,u,DB in safe_s]
    DB_list =[DB for danger_s, safe_s in samples for x,u,DB in safe_s]

    ## Solve the SVM optimization
    class_weight = {-1:1, 1:1} if class_weight is None else class_weight
    
    X_aug = kernel.augment(X)
    y = np.array(y).reshape((-1))
    lenw = int((dim+1)*dim/2 + dim + 1)
    print("X_aug shape", X_aug.shape)
    print("y shape", y.shape)

    feasiblePoints = [(x,xa) for x,xa,y in  zip(X,X_aug,y) if y ==1]
    feasiblePoints = [(x,xa,u,DA,DB,Dg) for (x,xa),u,(DA,DB,Dg) in zip(feasiblePoints,u_list,DB_list)]
    udim = len(u_list[0])
    print("udim:",udim)
    lenfu = len(feasiblePoints) * udim # The length of all the `du`
    uvec = np.concatenate(u_list,axis = 0) # make the array of u a vector
    assert(lenfu == len(uvec))
    print("len(feasiblePoints)",len(feasiblePoints))

    def obj(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return gamma0*sqeuclidean(w[:-1]) + gamma * np.sum(c) + gamma2 * sqeuclidean(u_ - uvec)

    def grad(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        ans = gamma0*2*w
        ans[-1] = 0
        ans = np.array(list(ans)+list([gamma]*len(c)) +list(2*gamma2*(u_ - uvec)))
        return ans.reshape((-1)) # shape (-1) rather than (1,-1) is necessary for trust-constr

    def SVMcons(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return (y * (X_aug @ w) - 1 + c).reshape(-1)
    def SVMjac(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return np.concatenate([y.reshape((-1,1))*X_aug, np.eye(len(c)),np.zeros((len(c),lenfu)) ],axis=1)

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

    def containCons(w):
        """
        The constraint that the new CBF should be contained by the old CBF
            This constraint is enforced by letting all points on the axis be inside of the old CBF
        """
        A,b,c = kernel.GetParam(w[:lenw])
        points,E,_c = representEclips(A,b,c)
        return np.array([p.T@HA@p + Hb.T@p + Hc for p in points]+list(E)+[_c])


    # # TODO
    # def symmetricCons(w):
    #     return

    options = {"maxiter" : 500, "disp"    : True,  "verbose":1, 'iprint':2} # 'iprint':2,
    lenx0 = lenw + len(y) + lenfu
    x0 = np.random.random(lenx0) *0
    # set the init value to be an eclipse around the mean of feasible points
    x0[[int((41-i)*i/2) for i in range(dim)]] = -1
    pos_x_mean = np.mean([x for x,y in zip(X,y) if y==1], axis = 0)
    x0[lenw-dim-1:lenw-1] = 2*pos_x_mean
    x0[lenw-1] = 1 - sqeuclidean(pos_x_mean) # set the c to be 1, so that the A should be negative definite
    x0[lenw:-lenfu]  *= 0 # set the init of c to be zero
    x0[-lenfu:] = uvec # set the init of u_ to be u

    constraints = [{'type':'ineq','fun':SVMcons, "jac":SVMjac},
                   {'type':'ineq','fun':feasibleCons, "jac":feasibleJac},
                   {'type':'ineq','fun':containCons}, # TODO decide whether to comment out this line
                   ]
    
    bounds = np.ones((lenx0,2)) * np.array([[-1,1]]) * 9999
    bounds[0,:] *= 0 # the first dim `x`  TODO the first dim of x should have more
    bounds[lenw:-lenfu,0] = 0 # set c > 0
    bounds[lenw:-lenfu,1] = np.inf
    bounds[lenw+np.array([i for i,y in enumerate(y) if y==1]),1] = 0  # TODO decide whether to comment out this line
    bounds[-lenfu:,0] = -np.array(list(GP.MAXTORQUE)*len(feasiblePoints))
    bounds[-lenfu:,1] =  np.array(list(GP.MAXTORQUE)*len(feasiblePoints))

    # ################################
    # ###Temporary Constrinats########
    # ################################
    # w_inds = set((list(range(lenw-1))))
    # w_inds.remove(74)
    # w_inds.remove(84)
    # w_inds.remove(189)
    # w_inds.remove(int((dim+1)*dim/2) + 4)
    # w_inds.remove(int((dim+1)*dim/2) + 14)
    # bounds[list(w_inds),:] *= 0 # set the value not related to q and dq to zero

    res = minimize(obj, x0, options = options,jac=grad, bounds=bounds,
                constraints=constraints, method =  'SLSQP') # 'trust-constr' , "SLSQP"

    # print("SVMcons(res.x) :",SVMcons(res.x))
    # assert((SVMcons(res.x)>-1e-2).all())
    # print("SVM Constraint:\n", SVMcons(res.x[:len(x0)]))
    return (*kernel.GetParam(res.x[:lenw],dim=dim), res, samples)
    # return (*kernel.GetParam(x0[:lenw],dim=dim), res)

 


class LearnCBFSession_t(Session):
    """
        define the Session class to record the information
    """
    pass

class LearnCBFSession(LearnCBFSession_t):
    def __init__(self, CBF0, name = "tmp", Iteras = 10, mc = 0.01, gamma0=0.01, gamma = 1, gamma2=1, class_weight = None, 
                    numSample = 200, dangerDT=0.01, safeDT=0.5, ProcessNum = None):
        super().__init__(expName="IOWalkLearn", name = name, Iteras = 10, mc = mc, gamma0 = gamma0, gamma = gamma, gamma2 = gamma2, class_weight = class_weight, 
                    numSample = numSample,dangerDT = dangerDT,safeDT = safeDT, ProcessNum = ProcessNum)

        self.Iteras = Iteras
        self.mc = mc
        self.gamma0 = gamma0
        self.gamma = gamma
        self.gamma2 = gamma2
        self.class_weight = class_weight
        self.numSample = numSample
        self.dangerDT = dangerDT
        self.safeDT = safeDT
        self.CBF0 = CBF0
        ProcessNum = max(1,multiprocessing.cpu_count() - 2) if ProcessNum is None else ProcessNum
        self.ProcessNum = ProcessNum
    
        self.resultPath = "data/learncbf/%s_%s"%(name,self._storFileName)
        os.makedirs(self.resultPath, exist_ok=True)
        self.add_info("resultPath",self.resultPath)

        self.IterationInfo_ = [] # a list of dict: {start_time, end_time, sampleFile, CBFFile}
        

    def body(self):
        CTRL.restart()
        NoKeyboardInterrupt = True
        dumpJson(*self.CBF0,os.path.join(self.resultPath,"CBF0.json"))
        currentCBF = self.CBF0
        with Pool(self.ProcessNum) as pool:
            for i in range(self.Iteras):
                assert NoKeyboardInterrupt, "KeyboardInterrupt caught"    
                try:
                    currentCBFFile = os.path.join(self.resultPath,"CBF%d.json"%i)
                    self.IterationInfo_.append({"start_time" : str(datetime.datetime.now()),
                                                "inputCBFFile": currentCBFFile})
                    # read from the current CBF file to avoid mistake
                    Polyparameter = json.load(open(currentCBFFile,"r")) 
                    CBF = (np.array(Polyparameter["A"]), np.array(Polyparameter["b"]), np.array(Polyparameter["c"]))
                    A,b,c, res, samples =  learnCBFIter(CBF, [ ], mc = self.mc, dim = 20, gamma0 = self.gamma0, gamma = self.gamma, gamma2 = self.gamma2, numSample = self.numSample, dangerDT = self.dangerDT, safeDT = self.safeDT, pool = pool)
                    self.IterationInfo_[-1]["Optimization_TerminationMessage"] = res.message
                    sampleFile = os.path.join(self.resultPath,"samples%d.json"%i)
                    self.IterationInfo_[-1]["sampleFile"] = sampleFile
                    pkl.dump(samples,open(sampleFile,"wb"))

                    currentCBFFile = os.path.join(self.resultPath,"CBF%d.json"%(i+1))
                    dumpJson(A,b,c,currentCBFFile)            
                    self.IterationInfo_[-1]["outputCBFFile"] = currentCBFFile
                    self.IterationInfo_[-1]["end_time"] = str(datetime.datetime.now())

                except KeyboardInterrupt as ex:
                    NoKeyboardInterrupt = False
            
    @LearnCBFSession_t.column
    def IterationInfo(self):
        return self.IterationInfo_


if __name__ == '__main__':
    s = LearnCBFSession(CBF_GEN_conic(10,10000,(0,1,0.1,4)),
        name = "redLegQ2")
    s()
