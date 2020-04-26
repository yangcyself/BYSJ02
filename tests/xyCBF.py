"""
Test and exhibit the framework of the CBF controller, the CBF is a quad form constraining the x,y. And CLF is dragging the torso angle
"""

import sys
sys.path.append(".")
from ctrl.CBFCtrl import *
from util.visulization import QuadContour
import matplotlib.pyplot as plt

Traj = []
def callback_traj(obj):
    Traj.append([obj.t] + list(obj.state[:7]))

if __name__ == "__main__":
    CBF_CTRL.restart()
    ct = CBF_CTRL()
    # The hand designed CBF for torso position
    mc = 10
    HA_CBF = np.zeros((14,14))
    HA_CBF[1,1] = mc
    HA_CBF[1,8] = HA_CBF[8,1] = 1

    # CBF of the body_x
    HA_CBF[0,0] = -mc * 50
    HA_CBF[0,7] = HA_CBF[7,0] = -1 * 50
    Hb_CBF = np.zeros(14)
    Hc_CBF = -mc * 0.5 * 0.5

    ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF)

    ## Add CLF of the torso angle
    ang = 0.2
    mc2 = 1
    HA = np.zeros((14,14))
    HA[2,2] = 1
    HA[9,9] = mc2 * mc2
    HA[2,9] = HA[9,2] = mc2
    Hb = np.array([0,0,-ang,0,0,0,0,  0,0,-mc2*ang,0,0,0,0])
    Hc = ang*ang
    ct.addCLF(HA,Hb,Hc)

    ct.callbacks.append(callback_traj)

    # time.sleep(10)
    ct.step(10,sleep=0)


    Traj = np.array(Traj)
    plt.plot(Traj[:,1], Traj[:,2], label = "xy trajectory")

    pts = QuadContour(HA_CBF,Hb_CBF,Hc_CBF,np.arange(-0.1,0.1,0.005))
    # pts = QuadContour(HA,Hb,Hc,[-0.001])
    # print(pts)
    plt.plot(pts[:,0], pts[:,1], ".", label = "CBF")
    plt.ylim(ymin = 0)
    plt.title("x,y trajectory with CBF")
    plt.legend()
    plt.show()

    plt.plot(Traj[:,0],Traj[:,2], label = "torso angle")
    plt.plot(Traj[:,0],[ang]*len(Traj), label = "0.2")
    plt.legend()
    plt.title("torso angle v.s time")
    plt.show()