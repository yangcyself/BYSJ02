"""
The more standard WBC controller

The main aim of designing this is to test the dynamics of other control modules
"""
# import numpy as np
import cvxpy as cp
import sys
sys.path.append(".")
from ctrl.rabbitCtrl import *


class WBC_CTRL(CTRL):
    
    def __init__(self):
        super().__init__()    
        WBC_CTRL.WBC.reg(self,kp = np.array([50,50,50]))
        WBC_CTRL.cmdFr.reg(self)
        self.torso_des = np.array([0,0.8,0])

    @CTRL_COMPONENT
    def WBC(self, kp = np.array([5000,5000,5000]), kd = np.array([20, 20, 20])):
        """
            Run a simple WBC control
            self.torso_des -> (Nsupport, Gmap, wrench_des, )
        """
        torso_des = self.torso_des
        torso_state = self.state[:3]
        dtorso_state = self.state[7:7+3]
        
        # Note that currently the gc_pos have an error of 0.3 caused by the sphere of the toe
        gc_index = np.array([3,6])[self.gc_mask]
        gc_pos = self.CoMs[gc_index] - self.CoMs[[0]]

        wrench_des = kp * (torso_des - torso_state) - kd * dtorso_state
        wrench_des += [0, -GP.GRAVITY * 32, 0]
        Nsupport = sum(self.gc_mask)
        if(Nsupport):
            Gmap = np.concatenate([np.tile(np.eye(2),(1,Nsupport)),
                np.cross(np.tile(gc_pos,(1,2)).reshape(-1,2),
                        np.tile(np.eye(2),(Nsupport,1)),axis = 1)[None,...]],axis = 0)
        else:
            Gmap = None
        return (Nsupport, Gmap, wrench_des)

        # Fr = torso_des

    @CTRL_COMPONENT
    def WBC_FR(self):

        (Nsupport, Gmap, wrench_des) = self.WBC
        if(Nsupport):
            Fr = np.linalg.pinv(Gmap) @ wrench_des
        else:
            Fr = np.array([])

        if(not self.gc_mask[0]):
            Fr = np.concatenate([np.zeros(2), Fr], axis = 0)
        if(not self.gc_mask[1]):
            Fr = np.concatenate([Fr,np.zeros(2)], axis = 0)
        return Fr


    @CTRL_COMPONENT
    def cmdFr(self):
        """
        self.WBC -> setJointTorques()
        
        Excert the desFr from WBC
        """
        J_toe = self.J_toe
        tor = -J_toe.T @ self.WBC_FR
        return self.setJointTorques(tor[3:])
        

class QP_WBC_CTRL(WBC_CTRL):

    def __init__(self):
        super().__init__()    
        QP_WBC_CTRL.WBC.reg(self,kp = np.array([50,50,50]))
        WBC_CTRL.cmdFr.reg(self)

    @CTRL_COMPONENT
    def WBC_FR(self, cpw=[10,10,1],cplm = 0.001,mu = 0.4):

        (Nsupport, Gmap, wrench_des) = self.WBC
        if(Nsupport):
            cpFc = cp.Variable(shape=2*Nsupport)
            cpWeights = np.diag(np.array(cpw))
            cpGmap, cpwrench = cpWeights @ Gmap , cpWeights @ wrench_des
            cpobj = cp.Minimize(cp.norm(cpGmap @ cpFc - cpwrench)**2 + cplm * cp.norm(cpFc)**2)
            cpcons = (
                    [0 <= cpFc[i*2+1] for i in range(Nsupport)]
                    + [-cpFc[i*2] <= mu * cpFc[i*2+1] for i in range(Nsupport)] 
                    + [cpFc[i*2] <= mu * cpFc[i*2+1] for i in range(Nsupport)]
                    )
                
            prob = cp.Problem(cpobj, cpcons)
            prob.solve(verbose=False)
            Fc = cpFc.value
        else:
            Fc = np.array([])

        if(not self.gc_mask[0]):
            Fc = np.concatenate([np.zeros(2), Fc], axis = 0)
        if(not self.gc_mask[1]):
            Fc = np.concatenate([Fc,np.zeros(2)], axis = 0)
        
        return Fc
