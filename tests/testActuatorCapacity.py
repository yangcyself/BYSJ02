"""
the aim of test is to plot the full capacity of some joints.
"""

from ctrl.rabbitCtrl import *
from learnCBF.IOWalk.IOWalkUtil import *
class OutsideTorqueCtrl(CTRL):
    @CTRL_COMPONENT
    def TimedActuate(self,func = None):
        assert func is not None, "a call back function is needed in `actuate` of outsideTorqueCtrl"
        return self.setJointTorques(func(self.t))
    
ct = OutsideTorqueCtrl()
ct.restart()

initstate[1] = 1
# initstate[4]

ct.setState(initstate)
holoright = p.createConstraint(GP.floor, -1, GP.robot, 5, p.JOINT_FIXED, 
            jointAxis = [0,1,0],parentFramePosition = [0,0,0.2349], 
            childFramePosition = [0,0,0.0],
            childFrameOrientation=[0,0,1,0])

ct.state[4]

# OutsideTorqueCtrl.TimedActuate.reg(ct,func= (lambda t:[0,0,100,0]))
ct.step(0.15)