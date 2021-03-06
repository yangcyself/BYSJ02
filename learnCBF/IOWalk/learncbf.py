import sys
import os
sys.path.append(".")
import numpy as np
import globalParameters as GP
GP.GUIVIS = False
GP.CATCH_KEYBOARD_INTERRPUT = False
from scipy.linalg import sqrtm,expm
from scipy.optimize import minimize
import cvxpy as cp
from learnCBF.IOWalk.IOWalkUtil import *
from learnCBF.FittingUtil import sqeuclidean, dumpCBFsJson, loadCBFsJson
from learnCBF.FittingUtil import kernel_Poly1 as kernel
from learnCBF.SampleUtil import randomSample
from util.CBF_builder import *
from ExperimentSecretary.Core import Session
import multiprocessing
from multiprocessing import Pool
import datetime
import time
from glob import glob
import dill as pkl
import json

# SYMMETRY_AUGMENT = False # deprecated, to use this, back to version [firstEditionPaper]

Balpha = loadCSV("./data/Balpha/ta_gait_design.csv")
samplerCallback = lambda args:None

def sampler(CBFs, mc, BalphaStd, Balpha = Balpha, CBFList = None):
    """
    sample a lot of trajectories, stop when the CBF constraint cannot be satisfied
        return sampled trajectories
    """
    from ctrl.CBFWalker import CBF_WALKER_CTRL, CTRL
    CTRL.restart()

    ct = CBF_WALKER_CTRL()
    reset(ct)
    Balpha = Balpha * (1 - BalphaStd/2 + BalphaStd*np.random.random(size = Balpha.shape))
    Kp = 200 + 1800*np.random.rand()
    Kd = np.log(Kp)*(0.5+np.random.rand())
    CBF_WALKER_CTRL.IOcmd.reg(ct,Balpha = Balpha)
    CBF_WALKER_CTRL.IOLin.reg(ct,kp = Kp, kd = Kd)
    CBF_WALKER_CTRL.CBF_CLF_QP.reg(ct,mc_b = mc)
    # CBFList = [] if CBFList is None else CBFList
    # [ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF) for HA_CBF,Hb_CBF,Hc_CBF in CBFList]
    [ct.addCBF(*CBF) for CBF in CBFs]
    Traj = []
    def callback_traj(obj):
        Traj.append((obj.t, obj.state, obj.Hx, obj.CBF_CLF_QP, (obj.DHA, obj.DHB, obj.DHg), obj.LOG_CBF_ConsValue))
    ct.callbacks.append(callback_traj)
    def callback_break(obj):
        return not (all(obj.LOG_CBF_ConsValue>-1e-6))
    ct.callbacks.append(callback_break)
    ct.step(30)
    samplerCallback(locals())
    return Traj,(Balpha,Kp,Kd)


def GetPoints(traj, CBFs, mc, dangerDT, safeDT, lin_eps):
    """
    get safe points and danger points given a trajectory.
        The safe points are the points before `safeDT` of the termination
        The danger points are the points from the opt problem (QP)
            min_{u,x} || x - x_0 ||
            s.t. x^+ = DA x + DB u + Dg
                 B(x^+) >= 0        \\forall B (locally linearlized)
                 mc*B(x^+) + dB(x^+,u) >=0  \\forall B (locally linearlized)
            The solution x* has to be within a certain limit from x_0, 
            otherwise it means all points near by are bad points.
    args lin_eps: the epsilon to bound the area of linearlization, used to limit x_danger+
    """
    # assert not (traj[-1][5]>0).all(), "NO CBF Cons is violated in the last step, traj might be convergent"
    if((traj[-1][5]>0).all()):
        x_danger = []
        x_safe = [(traj[i][2],traj[i][3],traj[i][4]) for i in np.random.choice(len(traj)-int(10*safeDT/GP.DT),1)],

    # the danger points
    DcA,DcB,Dcg = traj[-1][4] # the Continues dynamics of the terminal point
    tt,ts,tx,tu,tDc,tConsV = traj[-1] # the time, state, augmented state, inputu, CBFConsValue of the terminal state
    if(tt<min(2,10*safeDT)):
        print("traj too short, tt: %f < %f"%(tt,min(2,10*safeDT)))
        return [],[]
    # Note, the `-DT` means that the time goes backward
    Dcg = Dcg.reshape(-1,1)
    sysdt = expm(np.concatenate([ dangerDT * np.concatenate([DcA, DcB, Dcg],axis = 1),np.zeros((DcB.shape[1]+1, DcA.shape[1] + DcB.shape[1] + 1))], axis = 0))
    DA,DB,Dg = sysdt[:DcA.shape[0],:DcA.shape[1]], sysdt[:DcA.shape[0],DcA.shape[1]:-1], sysdt[:DcA.shape[0],-1:] # The dynamics with the time goes forwards
    sysdt_half = expm(np.concatenate([ dangerDT * np.concatenate([DcA, DcB, Dcg],axis = 1),np.zeros((DcB.shape[1]+1, DcA.shape[1] + DcB.shape[1] + 1))], axis = 0))
    DA_half,DB_half,Dg_half = sysdt_half[:DcA.shape[0],:DcA.shape[1]], sysdt_half[:DcA.shape[0],DcA.shape[1]:-1], sysdt_half[:DcA.shape[0],-1:] # The dynamics with the time goes forwards
    sysdt_back = expm(np.concatenate([ - dangerDT * np.concatenate([DcA, DcB, Dcg],axis = 1),np.zeros((DcB.shape[1]+1, DcA.shape[1] + DcB.shape[1] + 1))], axis = 0))
    DA_back,DB_back,Dg_back = sysdt_back[:DcA.shape[0],:DcA.shape[1]], sysdt_back[:DcA.shape[0],DcA.shape[1]:-1], sysdt_back[:DcA.shape[0],-1:] # The dynamics with the time goes forwards
    Dg_back = Dg_back.reshape(-1)
    Dg_half = Dg_half.reshape(-1)
    Dg = Dg.reshape(-1)
    Dcg = Dcg.reshape(-1)
    
    x_danger = [(tx, tu, tDc)]
    u_danger_backs = []
    for a in np.arange(0,20,4):
        cpu = cp.Variable(shape=len(GP.MAXTORQUE))
        cpx = cp.Variable(shape=DcA.shape[0])
        cpx_plus = DA @ cpx + DB @ cpu + Dg
        cpx_half = DA_half @ cpx + DB_half @ cpu + Dg_half
        cpBs = [tx .T @ HA @ (2*cpx_plus - tx) + Hb.T @ cpx_plus  + Hc for HA,Hb,Hc in CBFs] # B(x_plus), linearlized the quad term
        cpBs_half = [tx .T @ HA @ (2*cpx_half - tx) + Hb.T @ cpx_half  + Hc for HA,Hb,Hc in CBFs] # B(x_plus), linearlized the quad term
        a = 2**a
        cpa = np.diag([a]*(DcA.shape[0]//2)+[1]*(DcA.shape[0]//2))
        cpobj = cp.Minimize(cp.norm(cpa @ (cpx - tx))**2)
        cpcons = (
                [tx.T @ HA @ (2*cpx - tx) + Hb.T @ cpx  + Hc >=0 for HA,Hb,Hc in CBFs]
                + [ cpB >=0 for cpB in cpBs]
                + [ cpB >=0 for cpB in cpBs_half]
                + [ (mc * cpB + (2*tx.T @ HA + Hb.T) @ (DcA @ cpx_plus + DcB @ cpu + Dcg)) >= 0  # mc*B(x^+) + dB(x^+,u) >=0
                    for cpB,(HA,Hb,Hc) in zip(cpBs,CBFs)]
                + [ (mc * cpB + (2*tx.T @ HA + Hb.T) @ (DcA @ cpx_half + DcB @ cpu + Dcg)) >= 0  # mc*B(x^+) + dB(x^+,u) >=0
                    for cpB,(HA,Hb,Hc) in zip(cpBs_half,CBFs)]
                + [cpu <= GP.MAXTORQUE] + [cpu >= -GP.MAXTORQUE]
                )
            
        prob = cp.Problem(cpobj, cpcons)
        prob.solve(verbose=False)
        assert prob.status=="optimal" , "The problem didnot solved optimaly"
        x_danger_star = cpx.value
        u_danger_star = cpu.value
        FoundViolatedCBF = False
        if(np.linalg.norm((tx - x_danger_star)[:len(x_danger_star)//2])>lin_eps):
            continue
        u_danger_backs.append(u_danger_star)
        if(np.linalg.norm(tx - x_danger_star)<1e-3):
            break   
        x_danger.append((x_danger_star,u_danger_star,(DA,DB,Dg)))
    if(len(u_danger_backs)):
        u_danger_back = np.mean(u_danger_backs,axis=0).reshape(-1)
        x_danger_back = DA_back @ tx + DB_back @ u_danger_back + Dg_back 
        if(np.linalg.norm(tx - x_danger_back)>lin_eps):
            x_danger_back = tx + lin_eps * (x_danger_back-tx)/np.linalg.norm(tx - x_danger_back)
        x_danger.append(( x_danger_back, u_danger_back,(DA,DB,Dg)))
        
    # The safe Points
    x_safe = [(traj[i][2],traj[i][3],traj[i][4]) for i in [int(-safeDT/GP.DT),int(-1.5*safeDT/GP.DT)]]
    x_safe += [(traj[i][2],traj[i][3],traj[i][4]) for i in np.random.choice(len(traj)-int(10*safeDT/GP.DT),1)]
    return x_danger,x_safe


def getAsample(inputarg):
    if(len(inputarg)==5): inputarg.append(None) # ExceptionHandel
    CBF,dangerDT,safeDT,mc,lin_eps, ExceptionHandel = inputarg
    traj,param = sampler(CBF, mc, BalphaStd = 0.03)
    try:
        x_danger,x_safe = GetPoints(traj,CBF, mc,dangerDT,safeDT,lin_eps)
    except KeyboardInterrupt as ex:
        raise ex
    except Exception as ex:
        if(ExceptionHandel) is not None:
            ExceptionHandel(traj,param,ex)
            print("Record Exception:", str(ex))
            x_danger,x_safe = [],[]
        else:
            raise ex
    return x_danger,x_safe


def learnCBFIter(CBFs, badpoints, mc, dim, gamma0, gamma, gamma2, numSample, dangerDT, safeDT, lin_eps, protectPoints, class_weight= None,
                pool = None, sampleExceptionHander = None):
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
    args protectPoints: The list of points that y should be one, however, no u_list is provided
    """

    # try: #DEBUGGING CLOUSE
        # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
        # samples = pkl.load(open("./data/learncbf/redLegQ2_2020-05-14-15_16_18/samples0.json","rb"))
        # print("loaded %s from %s"%("samples", "./data/learncbf/tmp.pkl"))
    # except FileNotFoundError as ex:
    if pool is None:
        samples = [getAsample((CBFs, dangerDT,safeDT, mc, lin_eps, sampleExceptionHander)) for i in range(numSample)]
    else:
        samples = pool.map(getAsample, [(CBFs,dangerDT,safeDT,mc, lin_eps, sampleExceptionHander)]*numSample)
    pkl.dump(samples,open("./data/learncbf/tmp.pkl","wb"))

    X = [x for danger_s, safe_s in samples for x,u,DB in danger_s+safe_s]
    y = [i for danger_s, safe_s in samples for i in ([-1]*len(danger_s)+[1]*len(safe_s))]
    u_list = [u for danger_s, safe_s in samples for x,u,DB in safe_s]
    DB_list =[DB for danger_s, safe_s in samples for x,u,DB in safe_s]
    gamma_list = np.array([gamma] * len(y) if class_weight is None else [gamma * class_weight[yi] for yi in y])
    ## Solve the SVM optimization
    class_weight = {-1:1, 1:1} if class_weight is None else class_weight
    
    X_aug = kernel.augment(X)
    y = np.array(y).reshape((-1))
    lenw = kernel.GetParamDimension(dim)
    print("X_aug shape", X_aug.shape)
    print("y shape", y.shape)

    feasiblePoints = [(x,xa) for x,xa,y in  zip(X,X_aug,y) if y ==1]
    feasiblePoints = [(x,xa,u,DA,DB,Dg) for (x,xa),u,(DA,DB,Dg) in zip(feasiblePoints,u_list,DB_list)]
    udim = len(u_list[0])
    leny = len(y) # the length of y without protectPoints
    print("udim:",udim)
    lenfu = len(feasiblePoints) * udim # The length of all the `du`
    uvec = np.concatenate(u_list,axis = 0) # make the array of u a vector
    assert(lenfu == len(uvec))
    print("len(feasiblePoints)",len(feasiblePoints))

    ### Add Protect Points to the dataset
    if(protectPoints is not None and len(protectPoints)):
        protect_X_aug = kernel.augment(protectPoints)
        X = np.concatenate([X,protectPoints],axis=0)
        X_aug = np.concatenate([X_aug,protect_X_aug],axis=0)
        y = np.array(list(y) + [1]*len(protectPoints))
    else:
        protectPoints = [] # used for `len(protectPoints)`

    def obj(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return gamma0*sqeuclidean(w[:-1]) +  np.sum(gamma_list * c) + gamma2 * sqeuclidean(u_ - uvec)
    def grad(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        ans = gamma0*2*w
        ans[-1] = 0
        ans = np.array(list(ans)+list(gamma_list) +list(2*gamma2*(u_ - uvec)))
        return ans.reshape((-1)) # shape (-1) rather than (1,-1) is necessary for trust-constr


    def SVMcons(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        c = np.array(list(c)+[0]*len(protectPoints))
        return (y * (X_aug @ w) - 1 + c).reshape(-1)
    def SVMjac(w):
        w,c,u_ = w[:lenw],w[lenw:-lenfu],w[-lenfu:]
        return np.concatenate([y.reshape((-1,1))*X_aug, 
                            np.concatenate([np.eye(len(c)),np.zeros((len(protectPoints),len(c)))],axis = 0),
                            np.zeros((len(y),lenfu))],axis=1) # note: this len(y) is different from leny

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

    options = {"maxiter" : 500, "disp"    : True,  "verbose":1, 'iprint':2} # 'iprint':2,
    lenx0 = lenw + leny + lenfu
    x0 = np.random.random(lenx0) *0
    # x0[[int((41-i)*i/2) for i in range(1,dim)]] = -1
    # pos_x_mean = np.mean([x for x,y in zip(X,y) if y==1], axis = 0)
    # pos_x_mean[0] = 0
    # x0[lenw-dim-1:lenw-1] = 2*pos_x_mean
    # x0[lenw-1] = 1 - sqeuclidean(pos_x_mean) # set the c to be 1, so that the A should be negative definite

    x0[lenw:-lenfu]  *= 0 # set the init of c to be zero
    x0[-lenfu:] = uvec # set the init of u_ to be u

    constraints = [{'type':'ineq','fun':SVMcons, "jac":SVMjac},
                   {'type':'ineq','fun':feasibleCons, "jac":feasibleJac}]
    
    bounds = np.ones((lenx0,2)) * np.array([[-1,1]]) * 9999
    bounds[0,:] *= 0 # the first dim `x`  TODO the first dim of x should have more [POLY1]
    bounds[lenw:-lenfu,0] = 0 # set c > 0
    bounds[lenw:-lenfu,1] = np.inf # bound the protected points, no relaxation are allowed
    # bounds[lenw+np.array([i for i,y in enumerate(y) if y==1]),1] = 0  # TODO decide whether to comment out this line
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
    def __init__(self, CBFs0, name = "tmp", Iteras = 10, mc = 100, gamma0=0.01, gamma = 1, gamma2=1, class_weight = None, 
                    numSample = 200, dangerDT=0.01, safeDT=0.5, lin_eps = 0.2, protectPoints = None, ProcessNum = None):
        ProcessNum = max(1,multiprocessing.cpu_count() - 2) if ProcessNum is None else ProcessNum 
        # ProcessNum = None # IMPORTANT!!! The Behavior of Pybullet in multiprocess has not been tested
        super().__init__(expName="IOWalkLearn", name = name, Iteras = 10, mc = mc, gamma0 = gamma0, gamma = gamma, gamma2 = gamma2, class_weight = class_weight, 
                    numSample = numSample,dangerDT = dangerDT,safeDT = safeDT, lin_eps = lin_eps, ProcessNum = ProcessNum, protectPoints = protectPoints)

        self.Iteras = Iteras
        self.mc = mc
        self.gamma0 = gamma0
        self.gamma = gamma
        self.gamma2 = gamma2
        self.class_weight = class_weight
        self.numSample = numSample
        self.dangerDT = dangerDT
        self.safeDT = safeDT
        self.CBFs0 = CBFs0
        self.lin_eps = lin_eps
        self.ProcessNum = ProcessNum
    
        self.resultPath = "data/learncbf/%s_%s"%(name,self._storFileName)
        self.protectPoints = pkl.load(open(protectPoints,"rb")) if protectPoints is not None else None
        os.makedirs(self.resultPath, exist_ok=True)
        self.add_info("resultPath",self.resultPath)

        self.IterationInfo_ = [] # a list of dict: {start_time, end_time, sampleFile, CBFFile}
        self.SampleExceptions_ = []
        self.sampleLengths_ = []

    def sampleExceptionHander(self,traj,param,ex):
        dumpname = os.path.abspath(os.path.join(self.resultPath,"ExceptionTraj%d.pkl"%time.time()))
        # obj.t, obj.state, obj.Hx, obj.CBF_CLF_QP, (obj.DHA, obj.DHB, obj.DHg), obj.LOG_CBF_ConsValue)
        pkl.dump({
            "t": [t[0] for t in traj],
            "state": [t[1] for t in traj],
            "Hx": [t[2] for t in traj],
            "u": [t[3] for t in traj],
            "CBFCons": [t[5] for t in traj],
            "param":param
        } ,open(dumpname,"wb"))
        self.SampleExceptions_.append({"TrajPath":dumpname,"Exception":str(ex)})

    def samplercallBack_(self,kwargs):
        self.sampleLengths_.append(kwargs["Traj"][-1][0])
        # pkl.dump({
        #     "t": [t[0] for t in kwargs["Traj"]],
        #     "state": [t[1] for t in kwargs["Traj"]],
        #     "Hx": [t[2] for t in kwargs["Traj"]],
        #     "u": [t[3] for t in kwargs["Traj"]],
        #     "CBFCons": [t[5] for t in kwargs["Traj"]],
        # } ,open("data/tmp/%i.pkl"%int(time.time()),"wb"))

    def body(self):
        NoKeyboardInterrupt = True
        dumpCBFsJson(self.CBFs0,os.path.join(self.resultPath,"CBF0.json"))
        # pool = None
        
        global samplerCallback
        samplerCallback = self.samplercallBack_
        
        for i in range(self.Iteras):
            assert NoKeyboardInterrupt, "KeyboardInterrupt caught"    
            try:
                currentCBFFile = os.path.join(self.resultPath,"CBF%d.json"%i)
                self.IterationInfo_.append({"start_time" : str(datetime.datetime.now()),
                                            "inputCBFFile": currentCBFFile})
                # read from the current CBF file to avoid mistake
                CBFs = loadCBFsJson(currentCBFFile)
                if(self.ProcessNum >=1):
                    with Pool(self.ProcessNum) as pool:
                        A,b,c, res, samples =  learnCBFIter(CBFs, [ ], mc = self.mc, dim = 16, gamma0 = self.gamma0, gamma = self.gamma, gamma2 = self.gamma2, numSample = self.numSample, dangerDT = self.dangerDT, safeDT = self.safeDT, lin_eps=self.lin_eps, pool = pool, protectPoints=self.protectPoints, sampleExceptionHander=self.sampleExceptionHander)
                else:
                    pool = None
                    A,b,c, res, samples =  learnCBFIter(CBFs, [ ], mc = self.mc, dim = 16, gamma0 = self.gamma0, gamma = self.gamma, gamma2 = self.gamma2, numSample = self.numSample, dangerDT = self.dangerDT, safeDT = self.safeDT, lin_eps=self.lin_eps, pool = pool, protectPoints=self.protectPoints, sampleExceptionHander=self.sampleExceptionHander)    
                self.IterationInfo_[-1]["Optimization_TerminationMessage"] = res.message
                sampleFile = os.path.join(self.resultPath,"samples%d.pkl"%i)
                self.IterationInfo_[-1]["sampleFile"] = sampleFile
                pkl.dump(samples,open(sampleFile,"wb"))

                self.IterationInfo_[-1]["averageTrajLength"] = np.mean(self.sampleLengths_)
                self.sampleLengths_ = []

                currentCBFFile = os.path.join(self.resultPath,"CBF%d.json"%(i+1))
                addedCBF = [(A,b,c)]
                dumpCBFsJson(CBFs + addedCBF,currentCBFFile)            
                self.IterationInfo_[-1]["outputCBFFile"] = currentCBFFile
                self.IterationInfo_[-1]["end_time"] = str(datetime.datetime.now())

            except KeyboardInterrupt as ex:
                NoKeyboardInterrupt = False
        
    @LearnCBFSession_t.column
    def IterationInfo(self):
        return self.IterationInfo_

    @LearnCBFSession_t.column
    def SampleExceptions(self):
        return self.SampleExceptions_

if __name__ == '__main__':
    # protectSamples = randomSample(pkl.load(open("data/learncbf/relabeling_2020-05-28-12_24_15/ExceptionTraj1590742999.pkl","rb")),
    #                         tauInd=7,num = 100,IncludeCollisionPoints = 80,period=7)
    # protectPath = "./data/protectPoints/ProtectSamples-6-1.pkl"
    # pkl.dump(protectSamples,open(protectPath,"wb"))
    s = LearnCBFSession([
                         CBF_GEN_conic(8,1000,(*roots2coeff(-1,-np.math.pi/4,np.math.pi/4),2)), # limit on the torso angle
                         CBF_GEN_conic(8,1000,(*roots2coeff(-1,3*np.math.pi/4,5*np.math.pi/4),[0,0,0,1,0.5,0,0,0])), # limit on the toe-angle
                         CBF_GEN_conic(8,1000,(*roots2coeff(-1,3*np.math.pi/4,5*np.math.pi/4),[0,0,0,0,0,1,0.5,0])), # limit on the toe-angle
                         ],
        name = "newReference", numSample=150, Iteras = 10, dangerDT=0.003, safeDT=0.02,
        class_weight={1:0.9, -1:0.1}, ProcessNum=0)
    s()
