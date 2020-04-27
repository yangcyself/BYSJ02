"""
Test the walking with Virtual constraint: using the IO linearization control scheme
"""

import sys
sys.path.append(".")
from ctrl.CBFWalker import *
import matplotlib.pyplot as plt

CBF_WALKER_CTRL.restart()
ct = CBF_WALKER_CTRL()

Balpha =  np.array([
    [-0.160827987125513,	-0.187607134811070,	-0.0674336818331384,	-0.125516015984117,	-0.0168214255686846,	-0.0313250368138578],
    [0.0873653371709151,	0.200725783096372,	-0.0300252126874686,	0.148275517821605,	0.0570910838662556,	0.209150474872096],
    [-0.0339476931185450,	-0.240887593110385,	-0.840294819928089,	-0.462855953941200,	-0.317145689802212,	-0.157521190052112],
    [0.207777212951830,	    0.519994705831362,	2.12144766616269,	0.470232658089196,	0.272391319931093,	0.0794930524259044] ])

Balpha[[0,2],:] += np.math.pi
# Balpha[:,:] *= -1 # Because the different hand direction with matlab
CBF_WALKER_CTRL.IOcmd.reg(ct,Balpha = Balpha)
CBF_WALKER_CTRL.IOLin.reg(ct,kp = 1000, kd = 50)

# add a CBF controlling The angle from stance leg to torso
if True: ## CBF of the stance toe
    mc = 1000
    # dmth = np.math.pi/18
    dmth = np.math.pi/6
    mth0 = np.math.pi + dmth/2

    HA_CBF = np.zeros((20,20))
    HA_CBF[7,7] = - mc
    HA_CBF[7,17] = HA_CBF[17,7] = -1

    Hb_CBF = np.zeros(20)
    Hb_CBF[17] = 2 * mth0
    Hb_CBF[7] = mc * 2 * mth0
    Hc_CBF = mc * (dmth **2 - mth0 **2)
    ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF)

if True: ## HA_CBF of the swing toe
    mc = 1000
    dmth = np.math.pi/6
    mth0 = np.math.pi + dmth/2
    ma = np.math.pi/12 # the factor acts on the tau
    dmth = dmth + ma

    HA_CBF = np.zeros((20,20))
    HA_CBF[8,8] = - mc
    HA_CBF[8,18] = HA_CBF[18,8] = -1
    Hb_CBF = np.zeros(20)
    Hb_CBF[18] = 2 * mth0
    Hb_CBF[8] = mc * 2 * mth0
    Hc_CBF = mc * (dmth **2 - mth0 **2)

    HA_CBF[9,9] = mc * ma**2
    HA_CBF[9,19] = HA_CBF[19,9] = ma**2 
    Hb_CBF[9] = mc * 2 * dmth
    Hb_CBF[19] =  2 * dmth

    ct.addCBF(HA_CBF,Hb_CBF,Hc_CBF)


Traj = []
def callback_traj(obj):
    Traj.append((obj.t, obj.Hx, obj.IOcmd))
def callback_clear():
    Traj = []

def IOcmd_plot(dim = 0):
    plt.plot([t[0] for t in Traj], [t[1][dim+3] for t in Traj], label = "State dimension%d"%dim)
    plt.plot([t[0] for t in Traj], [t[2][0][dim] for t in Traj], label = "command dimension%d"%dim)
    plt.legend()
    plt.show()
def CBF_plot(dim = 0):
    plt.plot([t[0] for t in Traj], [t[1][dim+7] for t in Traj], label = "CBF state dim%d"%dim)
    plt.ylim((np.math.pi - 0.6, np.math.pi + 0.6))
    plt.legend()
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
    
    ct.step(3)
    CBF_plot()
