"""
Test the walking with Virtual constraint: using the IO linearization control scheme

Change log:

Tag: firstEditionPaper:
    using the Balpha "./data/Balpha/original.csv"

2020-07-02:
    changed to the framework using ta_gait_design.csv


"""

import sys
sys.path.append(".")
from ctrl.IOLinearCtrl import *
import matplotlib.pyplot as plt
import pickle as pkl
from ExperimentSecretary.Core import Session

IOLinearCTRL.restart()
ct = IOLinearCTRL()

Balpha = loadCSV("./data/Balpha/ta_gait_design.csv")
initState = loadCSV("./data/Balpha/init-ta_gait_design.csv")
IOLinearCTRL.IOcmd.reg(ct,Balpha = Balpha)

"""
good pid parameters:
- kp = 400, kd = 20
- kp = 1000, kd = 50
"""
IOLinearCTRL.IOLin.reg(ct,kp = 400, kd = 20)

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
    ct.setState(initState)

reset()

if __name__ == "__main__":
    
    with Session(__file__) as s:
        ct.step(100)
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