"""
The more standard WBC controller

The main aim of designing this is to test the dynamics of other control modules
"""
# import numpy as np
import cvxpy as cp

from ctrl.rabbitCtrl import *


class WBC_CTRL(CTRL):
    
    def __init__(self):
        super().__init__()    
        WBC_CTRL.WBC.reg(self,kp = np.array([50,50,50]))
        WBC_CTRL.cmdFr.reg(self)

    torso_des = np.array([0,0.8,0.5])

    @CTRL_COMPONENT
    def WBC(self, kp = np.array([5,5,5]), kd = np.array([0.2, 0.2, 0.2])):
        """
            Run a simple WBC control
            self.torso_des -> torque
        """
        torso_des = self.torso_des
        torso_state = self.state[:3]
        dtorso_state = self.state[7:7+3]
        
        # Note that currently the gc_pos have an error of 0.3 caused by the sphere of the toe
        gc_pos = self.CoMs[[3,6]] - self.CoMs[[0]]

        wrench_des = kp * (torso_des - torso_state) - kd * dtorso_state
        wrench_des += [0, -GP.GRAVITY * 32, 0]

        Nsupport = 2
        Gmap = np.concatenate([np.tile(np.eye(2),(1,Nsupport)),
            np.cross(np.tile(gc_pos,(1,2)).reshape(-1,2),
                    np.tile(np.eye(2),(Nsupport,1)),axis = 1)[None,...]],axis = 0)
        

        Fr = np.linalg.pinv(Gmap) @ wrench_des
        return Fr
        # Fr = torso_des

    @CTRL_COMPONENT
    def cmdFr(self):
        """
        self.WBC -> setJointTorques()
        
        Excert the desFr from WBC
        """
        J_toe = self.J_toe
        tor = -J_toe.T @ self.WBC
        return self.setJointTorques(tor[3:])
        

class QP_WBC_CTRL(WBC_CTRL):

    def __init__(self):
        super().__init__()    
        QP_WBC_CTRL.WBC.reg(self,kp = np.array([50,50,50]))
        WBC_CTRL.cmdFr.reg(self)

    @CTRL_COMPONENT
    def WBC(self, kp = np.array([5,5,5]), kd = np.array([0.2, 0.2, 0.2]), cpw=[10,10,1],cplm = 0.001,mu = 0.4):

        torso_des = self.torso_des
        torso_state = self.state[:3]
        dtorso_state = self.state[7:7+3]
        
        # Note that currently the gc_pos have an error of 0.3 caused by the sphere of the toe
        gc_pos = self.CoMs[[3,6]] - self.CoMs[[0]]

        wrench_des = kp * (torso_des - torso_state) - kd * dtorso_state
        wrench_des += [0, -GP.GRAVITY * 32, 0]

        Nsupport = sum(self.gc_mask)
        Gmap = np.concatenate([np.tile(np.eye(2),(1,Nsupport)),
            np.cross(np.tile(gc_pos,(1,2)).reshape(-1,2),
                    np.tile(np.eye(2),(Nsupport,1)),axis = 1)[None,...]],axis = 0)


        #### START THE OPTIMIZATION PROBLEM ####
        ########################################

        cpFc = cp.Variable(shape=2*Nsupport)
        cpWeights = np.diag(np.array(cpw))
        cpGmap, cpwrench = cpWeights @ Gmap , cpWeights @ wrench_des
        cpobj = cp.Minimize(cp.norm(cpGmap @ cpFc - cpwrench)**2 + cplm * cp.norm(cpFc)**2)
        cpcons = (#[cpFc[i*3+2]>=0 for i in range(Nsupport)] + 
                # [cp.norm(cpFc[i*3:i*3+2])<= mu * cpFc[i*3+2]for i in range(Nsupport)]

                #  [cp.abs(cpFc[i*3]) <= mu * cpFc[i*3+2]for i in range(Nsupport)]
                # +[cp.abs(cpFc[i*3+1]) <= mu * cpFc[i*3+2]for i in range(Nsupport)]
                [0 <= cpFc[i*2+1] for i in range(Nsupport)]
                + [-cpFc[i*2] <= mu * cpFc[i*2+1] for i in range(Nsupport)] 
                + [cpFc[i*2] <= mu * cpFc[i*2+1] for i in range(Nsupport)]
                )
            
        prob = cp.Problem(cpobj, cpcons)
        prob.solve(verbose=False)
        Fc = cpFc.value
        return Fc
