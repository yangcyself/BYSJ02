"""
Test the dynamic of the CBF calculation, 
    namely: dHx = DHA @ Hx + DHB @ u + DHg
            dBF = dB_b @ u + dB_c
Run the recorded trajectory and compare the dynamic of BF
    2020-05-06 23:40:58 dHx passed, dBF passed, 走势一样，real有毛刺， dBF主要为ddHx, ddHx 比dHx大两个数量级, ddHx也有毛刺
        毛刺出现在 腿swing 切换到stance那一下 (毕竟那个碰撞我也没建模)
    2020-05-11 11:11:13 测试拟合出来的CBF，发现除了毛刺之外，在DBF和DHX还有持续偏差的情况
    例如，在0.58秒的时候， dBF计算出来是负的，但实际是正的
"""

import sys
sys.path.append(".")
from ctrl.CBFWalker import *
from ctrl.playBackCtrl import *
import pickle as pkl
import matplotlib.pyplot as plt
import json

class playBackWalkerCtrl(CBF_WALKER_CTRL, playBackCTRL):
    
    def __init__(self,traj,*args,**kwargs):
        super().__init__(traj,*args,**kwargs)
        self.traj = traj
        self.resetComponents()
        print("self Control Components:", self.ctrl_components)
        playBackCTRL.playback.reg(self)
        self.trajLength = traj["t"][-1]


Traj = []
def callback_traj(obj):
    #                0      1          2                                    3                 4           
    Traj.append((obj.t, obj.playback, (obj.Hx, obj.DHA, obj.DHB, obj.DHg), (obj.HF, obj.dHF), obj.LOG_SWICH_STANCE))
def callback_clear():
    Traj = []


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
playBackWalkerCtrl.restart()


# traj = pkl.load(open("D:\\yangcy\\UNVSenior\\Graduation\\GradProject\\RabbitEnv\\data\\Traj\\1588767897.pkl","rb"))
# traj = pkl.load(open("data/Traj/1589161354.pkl","rb"))

traj = pkl.load(open("data/Traj/1589207105.pkl","rb"))

ct = playBackWalkerCtrl(traj = traj)
ct.setState(init_state)
ct.addCBF(*CBF_GEN_conic(10,1,(0,1,0.1,4)))
ct.addCBF(*CBF_GEN_conic(10,1,(0,1,0.1,6)))
# Polyparameter = json.load(open("data/CBF/Feasible_2020-05-08-14_31_01.json","r"))
# HA_CBF,Hb_CBF,Hc_CBF = np.array(Polyparameter["A"]), np.array(Polyparameter["b"]), np.array(Polyparameter["c"])
# ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF)

ct.callbacks.append(callback_traj)

def drawSwichStancePoint(y = 0):
    switchPoints = [t[0] for i,t in enumerate(Traj[:-1]) if Traj[i][4] == (1-Traj[i+1][4])]
    ax = plt.gca()
    ax.plot(switchPoints,[0]*len(switchPoints),".", label = "switchPoints")
    # ax.plot()

def plotDHX(dim = 0):
    plt.figure()
    Hx = np.array([t[2][0] for t in Traj])
    dHx = (Hx[1:] - Hx[:-1])/GP.DT
    plt.plot([t[0] for t in Traj[:-1]], dHx[:,dim], label = "real")
    plt.plot([t[0] for t in Traj], [(t[2][1] @ t[2][0] + t[2][2] @ t[1] + t[2][3])[dim] for t in Traj], label = "Calculated")
    drawSwichStancePoint()
    plt.title("plotDHX dim%d"%dim)
    plt.legend()
    plt.grid()
    plt.draw()


def plotDBF(dim = 0):
    # Traj.append((obj.t, obj.playback, (obj.Hx, obj.DHA, obj.DHB, obj.DHg), (obj.HF, obj.dHF)))
    # dBF = dB_b @ u + dB_c
    plt.figure()
    BF = np.array([t[3][0] for t in Traj])
    dBF = (BF[1:] - BF[:-1])/GP.DT
    plt.plot([t[0] for t in Traj[:-1]], dBF[:,dim], label = "real")
    plt.plot([t[0] for t in Traj], [(t[3][1][dim][0] @ t[1] + t[3][1][dim][1]) for t in Traj], label = "Calculated")
    drawSwichStancePoint()
    plt.title("plotDBF dim%d"%dim)
    plt.legend()
    plt.grid()
    plt.draw()

def plotU(dim = 0):
    plt.figure()
    plt.plot([t[0] for t in Traj[:-1]], [t[1][dim] for t in Traj[:-1]] , label = "U")
    drawSwichStancePoint()
    plt.title("plotU dim%d"%dim)
    plt.legend()
    plt.grid()
    plt.draw()

if __name__ == "__main__":

    ct.step(0.2)
    # ct.playstate(1)
    plotDHX(dim=4)
    plotDHX(dim=6)
    plotDHX(dim=14)
    plotDHX(dim=16)

    plotDBF(dim=0)
    plotDBF(dim=1)

    plotU(dim = 4)
    plotU(dim = 6)
    plt.show()