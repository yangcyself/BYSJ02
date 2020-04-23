"""
Test the Rigid Body dynamics of the controller
    However, I didnot found an API to get the exact contact forces
    Thus, I can only test the dyanmics when the robot is flowting in the air :)

Compare the output of `ddq` and following `simddq`

    The result is satisfying when dt is 1e-5, 
        However, the RBD becomes not accurate when dt = 1e-4 or 1e-3
            I don't know whether this accuracy can be accepted
"""
import sys
sys.path.append(".")
import globalParameters as GP
GP.DT = 1e-4
from ctrl.rabbitCtrl import *

if __name__ == "__main__":
    CTRL.restart()
    ct = CTRL()
    ang = np.pi/15
    ct.setState( [0, 2, 0] + list(np.random.random(4 + 7)))
    dq = np.zeros(7)
    lastdq = np.zeros(7)
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


    t = 0
    torque = np.zeros(7)
    toeForce = np.zeros(6)
    while(t<5):
        torque +=  np.random.random(7)  *2
        torque *= np.array([0,0,0, 1,1,1,1]) # because the first dimension has no motor

    
        ct.resetFlags()

        # toeForce = np.random.random(3) * 10 * np.array([0,1,1]) # the x dim is fixed
        toeForce += np.random.random(6)  * 1 * np.array([1,0,1,1,0,1])  # the x dim is fixed
        p.applyExternalForce(GP.robot,5, toeForce[:3] ,[0]*3, p.WORLD_FRAME)
        p.applyExternalForce(GP.robot,8, toeForce[3:] ,[0]*3, p.WORLD_FRAME)

        ct.setJointTorques(torque[3:])

        ddq = ct.RBD_A_inv @ (torque- ct.RBD_B + ct.J_toe.T @ toeForce[[0,2,3,5]])
        print("ddq   :", ddq)

        p.stepSimulation()

        lastdq = dq
        dq = ct.state[GP.QDIMS:]
        print("simddq:",  (dq - lastdq)/dt)
        t += dt