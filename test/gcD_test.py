"""
Test of the dynamics under the holonomic constraint of the ground contact constraints
"""

import sys
sys.path.append(".")
from rabbitCtrl import *

if __name__ == "__main__":
    CTRL.restart()
    ct = CTRL()
    
    ang = np.pi/15
    q_init = [0,0.8*np.cos(ang),0, np.pi - ang, 0, -(np.pi - ang), 0]
    ct.setState( q_init + list(np.random.random(7)))

    lastState = np.zeros(14)
    GP.MAXTORQUE = GP.MAXTORQUE.astype(float) *  np.inf
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
    
    holoright = p.createConstraint(GP.floor, -1, GP.robot, 5, p.JOINT_POINT2POINT, 
                        jointAxis = [1,0,0], parentFramePosition = [0,-0.3,0], 
                        childFramePosition = [0,0,0.03])

    hololeft = p.createConstraint(GP.floor, -1, GP.robot, 8, p.JOINT_POINT2POINT, 
                        jointAxis = [1,0,0], parentFramePosition = [0,0.3,0], 
                        childFramePosition = [0,0,0.03])

    t = 0
    torque = np.zeros(7)

    # ct.gc_mask = np.array([True,True])
    CTRL.gc_mask.reg(ct, constant = np.array([True,True]))

    while(t<5):
        torque +=  np.random.random(7)  *2
        torque *= np.array([0,0,0, 1,1,1,1]) * 0 # because the first dimension has no motor
        torque *= 0
    
        ct.resetFlags()
        ct.setJointTorques(torque[3:])


        dstate = ct.DA @ ct.state + ct.DB @ torque + ct.Dg
        print("dstate   :", dstate[GP.QDIMS:])

        p.stepSimulation()

        print("real:",  ((ct.state - lastState)/dt)[GP.QDIMS:])
        lastState = ct.state
        t += dt