"""
The controller of a rabbit pybullet environment, providing functions like restart, set state, and run controller
"""

import pybullet as p
import pybullet_data 
import os
import time
import globalParameters as GP
from globalParameters import qind
import numpy as np
import inspect
GRAVITY = -9.8


class CTRL_COMPONENT:
    def __init__(self,func):
        self.func = func
        self.name = self.func.__name__
        self.funcsig = [x.name for x in inspect.signature(self.func).parameters.values()]

    def __set__(self):
        raise ValueError("The control variable cannot be set")

    def __get__(self, obj, klass):
        if obj is None:
            return self
#             if(not getattr(obj,"%s__flag"%self.name)):
# #                  = self.func(obj,**self.params)
#                 setattr(obj,"%s__value"%self.name,self.func(obj,**getattr(obj,"%s__param"%self.name)))
#                 setattr(obj,"%s__flag"%self.name,True)
        if(not obj.ctrl_flags.get(self.name,False)):
            obj.ctrl_value[self.name] = self.func(obj,**(obj.ctrl_param.get(self.name,{})))
            obj.ctrl_flags[self.name] = True
        return obj.ctrl_value[self.name]
    
    def __call__(self, obj):
        return self.__get__(obj,type(obj))
    
    def reg(self,obj,**kwargs):
        """
            Note, modules still can be called even if it is not registered. 
            This function is only used to change the default parameters
        """
        obj.ctrl_components.append(self)
        assert (all([k in self.funcsig for k in kwargs.keys()]) or not 
            "argument not declared in control component %s"%self.func.__name__)
        obj.ctrl_param[self.name] = kwargs

    def resetFunc(self,obj,func):
        assert (self.name == func.__name__ or not
            "reset function should have the same name as the old one")

        self.func = func
class CTRL:
    """
        The framework of the controller is highly pythonic

        In each step of the simulation, The controller calls each registed controller components.

            The input and output of each control components are properties of the controller, or to directly calls pybullet API
                (fully side-effect)
            
            The variables are lazy functions so that the computation can be reused

            NOTE: Each Ctrl component accesses self.properties but cannot set any of those properties
    """

    def __init__(self):
        self.ctrl_components = []
        self.ctrl_flags = {}
        self.ctrl_param = {}
        self.ctrl_value = {}
        self.t = 0
    
        self.resetFlags()
        

    def _resetflag(self,k):
        self.ctrl_flags[k] = False


    def resetFlags(self):
        [self._resetflag(k) for k in self.ctrl_flags.keys()]


    def step(self,T = 1,sleep=None):
        for i in range(int(T/dt+0.5)):
            # Call all regested control components
            self.resetFlags()
            [c(self) for c in self.ctrl_components]    

            p.stepSimulation()
            if(sleep is not None):
                time.sleep(sleep)
            else:
                time.sleep(dt)
            self.t += dt
        return None

    @CTRL_COMPONENT
    def state(self):
        # return the current state in the format of [q, dq]
        return np.array([s[:2] for s in p.getJointStates(GP.robot,list(qind))]).T.reshape(-1)
    
    @CTRL_COMPONENT
    def CoMs(self):
        # return the Center of Mass of the links in the global frame
        #   [body, right_thigh, right_shin, right_toe, left_thign, left_shin, left_toe], body is ind 2
        linkInd = np.arange(2,9)
        return np.array([s[0][1:] for s in p.getLinkStates(GP.robot,linkInd)])

    @CTRL_COMPONENT
    def J_gc(self):
        J = [p.calculateJacobian(GP.robot,i,
            [0,0,0], # the point on the specified link
            list(self.state[:7]),
            [0 for a in range(7)], # joint velocities
            [0 for a in range(7)]) # desired joint accelerations
        for i in [5,8]] # calculates the linear jacobian and the angular jacobian
        # Each element of J is tuple (J_x, J_r), the jacobian of transition and rotation
        return np.concatenate([j[0][1:] for j in J], axis = 0)


    @staticmethod
    def setJointTorques(torque, mask = [1]*4):
        """
        arg torque: A four-element list 
        arg mask:   Whether to control this joint
        """
        for i,ind in enumerate(qind[3:]):
            if(mask[i]):
                p.setJointMotorControl2(GP.robot,ind,p.POSITION_CONTROL,0,force=0)
                p.setJointMotorControl2(bodyIndex=GP.robot,
                                        jointIndex=ind,
                                        controlMode=p.TORQUE_CONTROL,
                                        force=max(min(torque[i], GP.MAXTORQUE[i+3]),-GP.MAXTORQUE[i+3]))

    @staticmethod
    def setState(state):
        """
        Set the state of the robot, the state is either Qdim array assuming velocities are 0, or 2Qdim with velocities
        """
        state = np.array(state).reshape(-1)
        if(len(state) == GP.QDIMS):
            [p.resetJointState(robot,qind(i),t) for i,t in enumerate(state)]
        elif(len(state)==2*GP.QDIMS):
            dstate = state[len(state)//2:]
            state = state[:len(state)//2]
            [p.resetJointState(robot,qind(i),t,dt) for i,(t,dt)in enumerate(zip(state,dstate))]
        else:
            raise ValueError("State should be eigher QDIM or 2QDIM")


    @staticmethod
    def restart():
        global floor,robot,numJoints
        robot=floor=numJoints=None
        p.resetSimulation()
        p.setGravity(0,0,GRAVITY)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor = p.loadURDF("plane.urdf")

        # robotStartPos = [0, 0, -2+0.63]
        robotStartPos = [0, 0, 0]
        robotStartOrientation = p.getQuaternionFromEuler([0., 0, 0])
        robot = p.loadURDF("five_link_walker.urdf",robotStartPos,robotStartOrientation)

        numJoints = p.getNumJoints(robot)
        GP.robot, GP.floor, GP.numJoints = robot, floor, numJoints

        ang = np.pi/15
        q_init = [0,0.8*np.cos(ang),0, np.pi - ang, 0, -(np.pi - ang), 0]
        CTRL.setState(q_init)
        
        for j in range (GP.QDIMS):
            force=GP.MAXTORQUE[j]
            pos=q_init[j]
            p.setJointMotorControl2(robot,qind(j),p.POSITION_CONTROL,pos,force=force)
        
        for i in range(10):
            p.stepSimulation()




class WBC_CTRL(CTRL):

    def __init__(self):
        super().__init__()
        self.torso_des = np.array([0,0.8,0.5])
        WBC_CTRL.WBC.reg(self,kp = np.array([50,50,50]))
        WBC_CTRL.cmdFr.reg(self)

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
        wrench_des += [0, -GRAVITY * 32, 0]

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
        J_gc = self.J_gc
        tor = -J_gc.T @ self.WBC
        self.setJointTorques(tor[3:])
        return tor


robot=floor=numJoints=None
dt = 1e-3
# physicsClient = p.connect(p.DIRECT)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
#p.setRealTimeSimulation(True)
p.setGravity(0, 0, GRAVITY)
p.setTimeStep(dt)


if __name__ == "__main__":
    CTRL.restart()
    ct = WBC_CTRL()
    ct.step(10)