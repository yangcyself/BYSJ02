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



ct = CBF_WALKER_CTRL()

Balpha =  np.array([
    [-0.160827987125513,	-0.187607134811070,	-0.0674336818331384,	-0.125516015984117,	-0.0168214255686846,	-0.0313250368138578],
    [0.0873653371709151,	0.200725783096372,	-0.0300252126874686,	0.148275517821605,	0.0570910838662556,	0.209150474872096],
    [-0.0339476931185450,	-0.240887593110385,	-0.840294819928089,	-0.462855953941200,	-0.317145689802212,	-0.157521190052112],
    [0.207777212951830,	    0.519994705831362,	2.12144766616269,	0.470232658089196,	0.272391319931093,	0.0794930524259044] ])

init_state = np.array([
    -0.0796848040543580,
    0.795256332708201,
    0.0172782890029275,
    -0.160778382265038 + np.math.pi,
    0.0872665521987388,
    -0.0336100968515576 + np.math.pi,
    0.207826470055357,

    0.332565019796099,
    0.0122814228026706,
    0.100448183395960,
    -0.284872086257488,
    1.19997002232031,
    -2.18405395105742,
    3.29500361015889,
    ])


Balpha[[0,2],:] += np.math.pi
# Balpha[:,:] *= -1 # Because the different hand direction with matlab
CBF_WALKER_CTRL.IOcmd.reg(ct,Balpha = Balpha)
CBF_WALKER_CTRL.IOLin.reg(ct,kp = 500, kd = 20)

# add a CBF controlling The angle from stance leg to torso
Polyparameter = json.load(open("data/CBF/1_9weight_2020-05-03-20_53_09.json","r"))

# Make the init_state positive in CBF

ct.resetFlags()
IOLinearCTRL.state.set(ct,init_state)
IOLinearCTRL.contact_mask.set(ct,np.array((True,False)))
HA_CBF_,Hb_CBF_,Hc_CBF = np.array(Polyparameter["A"]), np.array(Polyparameter["b"]), np.array(Polyparameter["c"])
HA_CBF, Hb_CBF = np.zeros((20,20)), np.zeros(20)
HA_CBF[1:,1:] = HA_CBF_
Hb_CBF[1:] = Hb_CBF_
sign = np.sign(ct.Hx.T @ HA_CBF @ ct.Hx + Hb_CBF @ ct.Hx + Hc_CBF)
HA_CBF,Hb_CBF,Hc_CBF = sign * HA_CBF, sign * Hb_CBF, sign * Hc_CBF 

ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF)


Traj = []
def callback_traj(obj):
    Traj.append((obj.t, obj.state, obj.Hx, obj.IOcmd, obj.LOG_ResultFr, obj.CBF_CLF_QP, obj.LOG_FW))
def callback_clear():
    Traj = []

def IOcmd_plot(dim = 0):
    plt.plot([t[0] for t in Traj], [t[1][dim+3] for t in Traj], label = "State dimension%d"%dim)
    plt.plot([t[0] for t in Traj], [t[3][0][dim] for t in Traj], label = "command dimension%d"%dim)
    plt.legend()
    plt.show()


def CBF_plot():
    xx = [t[2] for t in Traj]
    plt.plot([t[0] for t in Traj],[x.T @ HA_CBF @ x + Hb_CBF @ x + Hc_CBF  for x in xx], label = "CBF")
    plt.plot([t[0] for t in Traj],[0 for x in xx], label = "0")
    plt.title("CBF")
    plt.show()

    
def Fr_plot():
    plt.plot([t[0] for t in Traj], [t[4][0] for t in Traj], label = "Fr state x")
    plt.plot([t[0] for t in Traj], [t[4][1] for t in Traj], label = "Fr state y")
    plt.plot([t[0] for t in Traj], [0 for t in Traj], label = "0")
    plt.grid()
    plt.legend()
    plt.show()

def U_plot(dim = 0):
    plt.plot([t[0] for t in Traj], [t[5][dim+3] for t in Traj], label = "torque dim %d"%dim)
    plt.legend()
    plt.grid()
    plt.show()

def fw_plot(dim=0):
    plt.plot([t[0] for t in Traj], [t[6][dim] for t in Traj], label = "Feedforward dim %d"%dim)
    plt.legend()
    plt.grid()
    plt.show()



ct.callbacks.append(callback_traj)
def callback_break(obj):
    return not (3*np.math.pi/4 < obj.Hx[7] < 5*np.math.pi/4 
                and 3*np.math.pi/4 < obj.Hx[8] < 5*np.math.pi/4
                and -0.1<obj.Hx[0])
ct.callbacks.append(callback_break)

def reset():
    ct.resetStatic()
    ct.setState(init_state)
CBF_WALKER_CTRL.restart()
reset()


if __name__ == "__main__":
    with Session(__file__) as s:
        ct.step(5)
        dumpname = os.path.abspath(os.path.join("./data/Traj","%d.pkl"%time.time()))
        pkl.dump({
            "t": [t[0] for t in Traj],
            "state": [t[1] for t in Traj],
            "u": [t[5] for t in Traj]
        } ,open(dumpname,"wb"))
        s.add_info("trajlog",dumpname)

    # IOcmd_plot(dim = 0)
    # IOcmd_plot(dim = 1)
    # IOcmd_plot(dim = 2)
    # IOcmd_plot(dim = 3)

    CBF_plot()
    # Fr_plot()
    
    # U_plot(dim = 0)
    # U_plot(dim = 1)
    # U_plot(dim = 2)
    # U_plot(dim = 3)

    # fw_plot(dim = 0)
    # fw_plot(dim = 1)
    # fw_plot(dim = 2)
    # fw_plot(dim = 3)