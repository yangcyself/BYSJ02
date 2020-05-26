"""
    This test the Generalizability of CBF. 
    The core of this test is to simulate and analysis the length of the trajectory, using different set of parameters and CBFs
"""

import sys
sys.path.append(".")
import globalParameters as GP
GP.GUIVIS = False
GP.CATCH_KEYBOARD_INTERRPUT = False
from ctrl.CBFWalker import *
import matplotlib.pyplot as plt
from ExperimentSecretary.Core import Session
from util.visulization import QuadContour
import pickle as pkl
import json
from learnCBF.FittingUtil import loadCBFsJson, loadJson
from util.CBF_builder import *

from glob import glob
from learnCBF.IOWalk.IOWalkUtil import reset as simuReset
experimentDir = "data/learncbf/protected_2020-05-25-01_40_21"

CBFSets = [loadCBFsJson(fileName=f) for f in glob(os.path.join(experimentDir,"CBF*.json"))]
CBFlens = [len(s) for s in CBFSets]
Trajs = [pkl.load(open(f,"rb")) for f in glob(os.path.join(experimentDir,"ExceptionTraj*.pkl"))]


def SimuTest(CBFs,Balpah,Kp,Kd):
    ct = CBF_WALKER_CTRL()
    ct.restart()
    simuReset(ct)
    CBF_WALKER_CTRL.IOcmd.reg(ct,Balpha = Balpah)
    CBF_WALKER_CTRL.IOLin.reg(ct,kp = Kp, kd = Kd)
    [ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF) for (HA_CBF,Hb_CBF,Hc_CBF) in CBFs] 

    Traj = []
    def callback_traj(obj):
    #                0      1          2       3          4                 5               6           7                      8                  9                  10
        Traj.append((obj.t, obj.state, obj.Hx, obj.IOcmd, obj.LOG_ResultFr, obj.CBF_CLF_QP, obj.LOG_FW, obj.LOG_CBF_ConsValue, obj.LOG_CBF_Value, obj.LOG_CBF_Drift, obj.LOG_dCBF_Value))    
    ct.callbacks.append(callback_traj)
    def callback_break(obj):
        return not (3*np.math.pi/4 < obj.Hx[7] < 5*np.math.pi/4 
                    and 3*np.math.pi/4 < obj.Hx[8] < 5*np.math.pi/4
                    and -0.1<obj.Hx[0])
    ct.callbacks.append(callback_break)
    ct.step(100)
    return Traj[-1][0]


def ACBF_Bparam(trajA,trajB):
    """
        test trajectory A with trajectory B's parameter
    """
    lenCBFs = len(trajA["CBFCons"][-1])
    CBFs = CBFSets[CBFlens.index(lenCBFs)]
    Balpha, Kp, Kd = trajB["param"]
    return SimuTest(CBFs,Balpha,Kp,Kd)

def randomParam(trajA):
    """
        test trajectory A with trajectory B's parameter
    """
    lenCBFs = len(trajA["CBFCons"][-1])
    CBFs = CBFSets[CBFlens.index(lenCBFs)]
    Balpha, Kp, Kd = trajA["param"]
    Balpha = Balpha + 0.02*(0.5-np.random.random(size = Balpha.shape))
    Kp = Kp + 50*(0.5-np.random.random())
    Kd = Kd + 10*(0.5-np.random.random())
    return SimuTest(CBFs,Balpha,Kp,Kd)



if __name__ == '__main__':
    result = {}
    with Session(__file__) as s:
        try:
            for i,traj in enumerate(Trajs):
                result[i] = {"ACBF_Bparam":{j:ACBF_Bparam(traj,traj_) for j,traj_ in enumerate(Trajs)},
                            # "randomParameters":[randomParam(traj) for j in range(10)]
                            }
        except AssertionError as ex:
            if("Interrupted"in str(ex)):
                pass
        finally:
            pkl.dump(result,open("data/analysis.pkl","wb"))
            s.add_info("resultPath","data/analysis.pkl")