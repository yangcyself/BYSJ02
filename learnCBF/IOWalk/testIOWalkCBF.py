"""
Test the walking with Virtual constraint: using the IO linearization control scheme
"""

import sys
sys.path.append(".")
import globalParameters as GP
# GP.STARTSIMULATION = False
from ctrl.CBFWalker import *
import matplotlib.pyplot as plt
from ExperimentSecretary.Core import Session
from util.visulization import QuadContour
import pickle as pkl
import json
from learnCBF.FittingUtil import loadCBFsJson, loadJson
from util.CBF_builder import *


ct = CBF_WALKER_CTRL()

Balpha = loadCSV("./data/Balpha/ta_gait_design.csv")
init_state = loadCSV("./data/Balpha/init-ta_gait_design.csv")


Kp, Kd = 400,20

CBF_WALKER_CTRL.IOcmd.reg(ct,Balpha = Balpha)
CBF_WALKER_CTRL.IOLin.reg(ct,kp = Kp, kd = Kd)


# add CBF to constraint on the leg
# ct.addCBF(*CBF_GEN_conic(10,100,(0,1,0.1,4)))
# ct.addCBF(*CBF_GEN_conic(10,100,(0,1,0.1,6)))
# ct.addCBF(*CBF_GEN_degree1(10,(0,1,-0.1,0))) # the x-velocity should be greater than 0.1

# ct.addCBF(*CBF_GEN_conic(10,100,(1,np.math.pi/3,(35*np.math.pi/180)**2-(np.math.pi/6)**2,4)))
# ct.addCBF(*CBF_GEN_conic(10,100,(1,np.math.pi/3,(35*np.math.pi/180)**2-(np.math.pi/6)**2,6)))
# ct.addCBF(*CBF_GEN_conic(10,10000,(-1,0,(np.math.pi/4)**2,2)))

Traj = []
def callback_traj(obj):
    #                0      1          2       3          4                 5               6           7                      8                  9                  10
    Traj.append((obj.t, obj.state, obj.Hx, obj.IOcmd, obj.LOG_ResultFr, obj.CBF_CLF_QP, obj.LOG_FW, obj.LOG_CBF_ConsValue, obj.LOG_CBF_Value, obj.LOG_CBF_Drift, obj.LOG_dCBF_Value))
def callback_clear():
    Traj = []

def IOcmd_plot(dim = 0):
    plt.figure()
    plt.plot([t[0] for t in Traj], [t[1][dim+3] for t in Traj], label = "State dimension%d"%dim)
    plt.plot([t[0] for t in Traj], [t[3][0][dim] for t in Traj], label = "command dimension%d"%dim)
    plt.legend()
    plt.draw()


def CBF_plot():
    plt.figure()
    xx = [t[2] for t in Traj]
    plt.plot([t[0] for t in Traj],[x.T @ HA_CBF @ x + Hb_CBF @ x + Hc_CBF  for x in xx], label = "CBF")
    plt.plot([t[0] for t in Traj],[0 for x in xx], label = "0")
    plt.title("CBF")
    plt.draw()

    
def Fr_plot():
    plt.figure()
    plt.plot([t[0] for t in Traj], [t[4][0] for t in Traj], label = "Fr state x")
    plt.plot([t[0] for t in Traj], [t[4][1] for t in Traj], label = "Fr state y")
    plt.plot([t[0] for t in Traj], [0 for t in Traj], label = "0")
    plt.grid()
    plt.legend()
    plt.draw()

def U_plot(dim = 0):
    plt.figure()
    plt.plot([t[0] for t in Traj], [t[5][dim+3] for t in Traj], label = "torque dim %d"%dim)
    plt.legend()
    plt.grid()
    plt.draw()

def fw_plot(dim=0):
    plt.figure()
    plt.plot([t[0] for t in Traj], [t[6][dim] for t in Traj], label = "Feedforward dim %d"%dim)
    plt.legend()
    plt.grid()
    plt.draw()

def CBFConsValue_plot(dim = None):
    plt.figure()
    if(type(dim)==int): dim = [dim] 
    elif dim is None: dim = range(len(Traj[0][7]))
    [plt.plot([t[0] for t in Traj], [t[7][d] for t in Traj], label = "CBF constraint value dim %d"%d) for d in dim]
    plt.title("CBFConsValue_plot")
    plt.legend()
    plt.grid()
    plt.draw()


def CBFvalue_plot(dim = None):
    plt.figure()
    if(type(dim)==int): dim = [dim]
    elif dim is None: dim = range(len(Traj[0][7]))
    [plt.plot([t[0] for t in Traj], [t[8][d] for t in Traj], label = "CBF value dim %d"%d) for d in dim]
    plt.title("CBFvalue_plot")
    plt.legend()
    plt.grid()
    plt.draw()

def CBFdvalue_plot(dim = None):
    plt.figure()
    if(type(dim)==int): dim = [dim]
    elif dim is None: dim = range(len(Traj[0][7]))
    [plt.plot([t[0] for t in Traj], [t[10][d] for t in Traj], label = "CBF dvalue dim %d"%d) for d  in dim]
    plt.title("CBFdvalue_plot")
    plt.legend()
    plt.grid()
    plt.draw()


def CBFDrift_plot(dim = 0):
    plt.figure()
    plt.plot([t[0] for t in Traj], [t[9][dim] for t in Traj], label = "CBF Drift dim %d"%dim)
    plt.title("CBFdrift_plot")
    plt.legend()
    plt.grid()
    plt.draw()



ct.callbacks.append(callback_traj)
def callback_break(obj):
    return False
    # return not (3*np.math.pi/4 < obj.Hx[7] < 5*np.math.pi/4 
    #             and 3*np.math.pi/4 < obj.Hx[8] < 5*np.math.pi/4
    #             and -0.1<obj.Hx[0])
ct.callbacks.append(callback_break)

def reset():
    ct.resetStatic()
    ct.setState(init_state)
CBF_WALKER_CTRL.restart()
reset()


if __name__ == "__main__":
    with Session(__file__) as s:
        ct.step(30)
        dumpname = os.path.abspath(os.path.join("./data/Traj","%d.pkl"%time.time()))
        pkl.dump({
            "t": [t[0] for t in Traj],
            "state": [t[1] for t in Traj],
            "Hx": [t[2] for t in Traj],
            "u": [t[5] for t in Traj]
        } ,open(dumpname,"wb"))
        s.add_info("trajlog",dumpname)

    # IOcmd_plot(dim = 0)
    # IOcmd_plot(dim = 1)
    # IOcmd_plot(dim = 2)
    # IOcmd_plot(dim = 3)

    ### Academic style plot
    # from util.AcademicPlotStyle import *
    # plt.plot([t[0] for t in Traj], [t[8][0] for t in Traj], label = "$B_{0\_safe}$")
    # plt.plot([t[0] for t in Traj], [t[8][1] for t in Traj], label = "$B_{1\_safe}$")
    # plt.xlabel("time($s$)")
    # plt.ylabel("CBF value")
    # plt.legend(**legendParam)
    # plt.grid()
    # CBF_plot()
    ### Academic style plot
    

    # CBF_plot()
    CBFConsValue_plot()#dim = [1,2])
    # CBFConsValue_plot(dim = 1)
    CBFvalue_plot()#dim = [1,2])
    # CBFvalue_plot(dim = 1)
    # CBFDrift_plot(dim = 0)
    # CBFDrift_plot(dim = 1)
    CBFdvalue_plot()#dim = [1,2])
    # CBFdvalue_plot(dim = 1)
    # Fr_plot()
    
    # U_plot(dim = 0)
    # U_plot(dim = 1)
    # U_plot(dim = 2)
    # U_plot(dim = 3)

    # fw_plot(dim = 0)
    # fw_plot(dim = 1)
    # fw_plot(dim = 2)
    # fw_plot(dim = 3)

    plt.show()