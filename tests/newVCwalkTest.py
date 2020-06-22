"""
Test the walking with reference bezier from TA_Gait_design
"""

import sys
sys.path.append(".")
from ctrl.IOLinearCtrl import *
import matplotlib.pyplot as plt
import pickle as pkl
from ExperimentSecretary.Core import Session
from util.BezierUtil import loadCSV,evalBezier

IOLinearCTRL.restart()
ct = IOLinearCTRL()


Balpha = loadCSV("./data/Balpha/ta_gait_design.csv")


ct.ptime= 0.6366
ct.thetaLength = 0.4608
ct.theta0 = 2.9085
# The new VC_c function
def VC_c(self):
    """
    (tau, s, ds, phase_len)
        s: The clock variable of the virtual constraint from one to zero
        Phase_len: The total time for the phase to go from 0 to 1, used to calculate the dt and ddt
            Here I use the x-axis of the robot
    """
    redStance, theta_plus, Labelmap = self.STEP_label
    q = Labelmap @ self.state[3:7]
    dq = Labelmap @ self.state[10:14]
    th = self.state[2]+q[0]+q[1]/2
    s = (th - self.theta0) / self.thetaLength;
    # s = s;
    return s, th, self.state[9]+dq[0]+dq[1]/2, self.thetaLength
IOLinearCTRL.VC_c.resetFunc(ct,VC_c)

def IOcmd(self,Balpha = None):
    """
        (Y_c, dY_c, ddY_c) where each cmd is a length-4 vector
    """
    tau, s, ds, phase_len = self.VC_c
    tau = max(min(tau,1),0)
    x,dx,ddx = evalBezier(Balpha,tau)
    redStance, theta_plus, Labelmap = self.STEP_label
    if(tau==0 or tau==1):
        dx *= 0
    return Labelmap@ x, Labelmap@ dx/self.ptime, Labelmap@ ddx *0 ### NOTE disable the feedforward of the acc of the gait
IOLinearCTRL.IOcmd.resetFunc(ct,IOcmd)
IOLinearCTRL.IOcmd.reg(ct,Balpha = Balpha)

IOLinearCTRL.IOLin.reg(ct,kp = np.array([2000,800,2000,800]), kd = np.array([50,50,50,50]))

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
    0.0002,
    0.7754,
    0.2033,
    2.6898,
    0.4973,
    2.5432,
    1.1158,
    
    0.5605,
   -0.0008,
    0.0243,
    0.6951,
    0.0070,
   -1.6232,
    0.2042
    ])

reset()

if __name__ == "__main__":
    
    # with Session(__file__) as s:
    #     ct.step(5)
    #     dumpname = os.path.abspath(os.path.join("./data/Traj","%d.pkl"%time.time()))
    #     pkl.dump({
    #         "t": [t[0] for t in Traj],
    #         "state": [t[1] for t in Traj],
    #         "u": [t[5] for t in Traj]
    #     } ,open(dumpname,"wb"))
    #     s.add_info("trajlog",dumpname)

    ct.step(100)
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