"""
Test the walking with Virtual constraint: using the IO linearization control scheme
"""

import sys
sys.path.append(".")
from ctrl.IOLinearCtrl import *
import matplotlib.pyplot as plt

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
IOLinearCTRL.IOLin.reg(ct,kp = 10000, kd = 200)

ct.step_t = 0.1576
ct.theta_minus =  0.0779
ct.theta_plus = -0.0797

Traj = []
def callback_traj(obj):
    Traj.append((obj.t, obj.state, obj.IOcmd))
def callback_clear():
    Traj = []
def callback_plot(dim = 0):
    plt.plot([t[0] for t in Traj], [t[1][dim+3] for t in Traj], label = "State dimension%d"%dim)
    plt.plot([t[0] for t in Traj], [t[2][0][dim] for t in Traj], label = "command dimension%d"%dim)
    plt.legend()
    plt.show()

ct.callbacks.append(callback_traj)

def reset():
    ct.setState([
    -0.0796848040543580,
    0.795256332708201,
    -1 * 0.0172782890029275,
    -1 * -0.160778382265038 + np.math.pi,
    -1 * 0.0872665521987388,
    -1 * -0.0336100968515576 + np.math.pi,
    -1 * 0.207826470055357,

    0.332565019796099,
    0.0122814228026706,
    -1 * 0.100448183395960,
    -1 * -0.284872086257488,
    -1 * 1.19997002232031,
    -1 * -2.18405395105742,
    -1 * 3.29500361015889,
    ])

reset()

if __name__ == "__main__":
    
    ct.step(10,sleep=0)
