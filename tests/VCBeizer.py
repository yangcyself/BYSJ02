"""
    Plot the Beizer curve of the IOLinearCtrller. 
        Compare this curve with the curve from matlab
    passed
"""

import sys
sys.path.append(".")
import globalParameters as GP
GP.STARTSIMULATION = False
from ctrl.IOLinearCtrl import *

import matplotlib.pyplot as plt

ct = IOLinearCTRL()

Balpha =  np.array([
    [-0.160827987125513,	-0.187607134811070,	-0.0674336818331384,	-0.125516015984117,	-0.0168214255686846,	-0.0313250368138578],
    [0.0873653371709151,	0.200725783096372,	-0.0300252126874686,	0.148275517821605,	0.0570910838662556,	0.209150474872096],
    [-0.0339476931185450,	-0.240887593110385,	-0.840294819928089,	-0.462855953941200,	-0.317145689802212,	-0.157521190052112],
    [0.207777212951830,	    0.519994705831362,	2.12144766616269,	0.470232658089196,	0.272391319931093,	0.0794930524259044] ])

Balpha[[0,2],:] += np.math.pi
# Balpha[:,:] *= -1 # Because the different hand direction with matlab
IOLinearCTRL.IOcmd.reg(ct,Balpha = Balpha)

C = np.arange(0.001,1,0.02)

def getBeizer(c):
    ct.resetFlags()
    IOLinearCTRL.VC_c.set(ct,c)
    return ct.IOcmd

cmds = [getBeizer((c, 1, 1, 0.1576)) for c in C] # pass s and ds as 1 to only compute the phi dphi and ddphi

phiarray = np.array([c[0] for c in cmds])
for i in range(phiarray.shape[1]):
    plt.plot(C,phiarray[:,i],label = "q%d"%(i+1))
plt.legend()
plt.show()