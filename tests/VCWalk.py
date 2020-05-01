"""
Test the walking with Virtual constraint: using the IO linearization control scheme
"""

import sys
sys.path.append(".")
from ctrl.IOLinearCtrl import *
import matplotlib.pyplot as plt
import pickle as pkl
from ExperimentSecretary.Core import Session

IOLinearCTRL.restart()
ct = IOLinearCTRL()

Balpha =  np.array([
    [-0.160827987125513,	-0.187607134811070,	-0.0674336818331384,	-0.125516015984117,	-0.0168214255686846,	-0.0313250368138578],
    [0.0873653371709151,	0.200725783096372,	-0.0300252126874686,	0.148275517821605,	0.0570910838662556,	0.209150474872096],
    [-0.0339476931185450,	-0.240887593110385,	-0.840294819928089,	-0.462855953941200,	-0.317145689802212,	-0.157521190052112],
    [0.207777212951830,	    0.519994705831362,	2.12144766616269,	0.470232658089196,	0.272391319931093,	0.0794930524259044] ])

Balpha[[0,2],:] += np.math.pi
# Balpha[:,:] *= -1 # Because the different hand direction with matlab
IOLinearCTRL.IOcmd.reg(ct,Balpha = Balpha)
IOLinearCTRL.IOLin.reg(ct,kp = 1000, kd = 50)

Traj = []

def callback_traj(obj):
    Traj.append((obj.t, obj.state, None, obj.IOcmd, None, obj.IOLin, obj.LOG_FW))
def callback_clear():
    Traj = []

def IOcmd_plot(dim = 0):
    plt.plot([t[0] for t in Traj], [t[1][dim+3] for t in Traj], label = "State dimension%d"%dim)
    plt.plot([t[0] for t in Traj], [t[3][0][dim] for t in Traj], label = "command dimension%d"%dim)
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

def reset():
    ct.resetStatic()
    ct.setState([
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

    IOcmd_plot(dim = 0)
    IOcmd_plot(dim = 1)
    IOcmd_plot(dim = 2)
    IOcmd_plot(dim = 3)

    # CBF_plot(dim = 0)
    # CBF_plot(dim = 1)
    # Fr_plot()
    
    U_plot(dim = 0)
    U_plot(dim = 1)
    U_plot(dim = 2)
    U_plot(dim = 3)

    fw_plot(dim = 0)
    fw_plot(dim = 1)
    fw_plot(dim = 2)
    fw_plot(dim = 3)