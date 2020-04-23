"""
Test of the dynamics under the holonomic constraint of the ground contact constraints
    Compare the dstate and the torque
However, this is not very accurate, But the error may be caused by the PD in the constraint of the dynamic
"""

import sys
sys.path.append(".")
import globalParameters as GP
GP.DT = 1e-3
from ctrl.WBCCtrl import *

if __name__ == "__main__":
    CTRL.restart()
    ct = WBC_CTRL()
    # ct = QP_WBC_CTRL()
    ct.torso_des = np.array([0,1.5,0.1])
    CTRL.gc_mask.reg(ct, constant = np.array([True,True]))

    ang = np.pi/15
    q_init = [0, 0.8*np.cos(ang) +0.05, 0, np.pi - ang, 0, -(np.pi - ang), 0]
    ct.setState( q_init + list(np.random.random(7)))

    lastState = np.zeros(14)
    for link_idx in qind:
        p.changeDynamics(GP.robot, link_idx, 
            linearDamping=0.0,
            angularDamping=0.0,
            jointDamping=0.0,
            lateralFriction = 0,
            spinningFriction = 0,
            rollingFriction = 0,
            anisotropicFriction = 0
        )
    
    parentFramePosition = np.array([0,0,0.1])
    parentFramePosition[GP.PRISMA_AXIS[0]] =  - GP.CROSS_SIGN * 0.3
    jointAxis = [1,1,0]
    jointAxis[GP.PRISMA_AXIS[0]] = 0
    holoright = p.createConstraint(GP.floor, -1, GP.robot, 5, p.JOINT_POINT2POINT, 
                        jointAxis = jointAxis, parentFramePosition = list(parentFramePosition), 
                        childFramePosition = [0,0,0.03])

    hololeft = p.createConstraint(GP.floor, -1, GP.robot, 8, p.JOINT_POINT2POINT, 
                        jointAxis = jointAxis, parentFramePosition = list(parentFramePosition*np.array([-1,-1,1])), 
                        childFramePosition = [0,0,0.03])

    t = 0
    torque = np.zeros(7)
    while(t<5):
        # torque +=  (np.random.random(7)-0.5)  *4
        # torque *= np.array([0,0,0, 1,1,1,1]) * 0 # because the first dimension has no motor
    
        ct.resetFlags()
        torque = ct.cmdFr # ct.cmdFr has the effect of setting torque
        # Fr = ct.WBC
        # print("Fr :", Fr)
        # ct.setJointTorques(torque[3:])
        print("torque :", torque)
        dstate = ct.DA @ ct.state + ct.DB @ torque + ct.Dg
        print("dstate   :", dstate[GP.QDIMS:])

        p.stepSimulation()

        print("real:",  ((ct.state - lastState)/dt)[GP.QDIMS:])
        lastState = ct.state
        t += dt