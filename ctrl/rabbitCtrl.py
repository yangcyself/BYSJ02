"""
The controller of a rabbit pybullet environment, providing functions like restart, set state, and run controller

The controllers in the 
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
        self.callbacks = []

    def _resetflag(self,k):
        self.ctrl_flags[k] = False


    def resetFlags(self):
        [self._resetflag(k) for k in self.ctrl_flags.keys()]


    def step(self,T = None,sleep=None):
        T = dt if T is None else T
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

            callres = [c(self) for c in self.callbacks]
            if(any(callres)): # use the call backs as break point checkers
                return callres
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
    def J_toe(self):
        J = [p.calculateJacobian(GP.robot,i,
            [0,0,0], # the point on the specified link
            list(self.state[:7]),
            [0 for a in range(7)], # joint velocities
            [0 for a in range(7)]) # desired joint accelerations
        for i in [5,8]] # calculates the linear jacobian and the angular jacobian
        # Each element of J is tuple (J_x, J_r), the jacobian of transition and rotation
        return np.concatenate([j[0][1:] for j in J], axis = 0)


    @CTRL_COMPONENT
    def contact_mask(self):
        """
        A boolean tuple representing whether the foot is interacting with the ground
        """
        contact_mask = np.array([len(p.getContactPoints(GP.robot,GP.floor,i)) for i in [5,8]]).astype(np.bool)
        return contact_mask

    @CTRL_COMPONENT
    def gc_mask(self,constant = None):
        """
        represening which toe need the ground contact constraint
            The default case is the contacting point. However, this may not be the case when the foot tries to lift
        """
        if constant is None:
            return self.contact_mask
        else:
            return constant


    @CTRL_COMPONENT
    def J_gc(self):
        """
        The jacobian of the toes that are gc
        """
        return self.J_toe[(np.ones((2,2)).astype(bool) * self.gc_mask).T.reshape(-1), ...]


    @CTRL_COMPONENT
    def RBD_A(self):
        """
        This is the Rigid Body Dynamics A matrix
         RBD_A ddq + RBD_B = torque + J.T F
        """
        return np.asarray(p.calculateMassMatrix(robot, list(self.state[:GP.QDIMS])))


    @CTRL_COMPONENT
    def RBD_A_inv(self):
        return np.linalg.inv(self.RBD_A)


    @CTRL_COMPONENT
    def RBD_B(self):
        """
        The is the non_linear_effects of the rigid body dynamics
            RBD_B ddq + RBD_B = torque + J.T F
        """
        return np.array(p.calculateInverseDynamics(robot, list(self.state[:GP.QDIMS]), list(self.state[GP.QDIMS:]), [0.0] * GP.QDIMS))


    @CTRL_COMPONENT
    def gc_Lam(self):
        """
        This is the lambda of the operational space of the gc
            Lambda = (J A^{-1} J^T)^-1
        """
        return np.linalg.inv(self.J_gc @ self.RBD_A_inv @ self.J_gc.T)


    @CTRL_COMPONENT
    def dJ_toe(self):
        """
            Time derivative of J_gc
        """
        sin = np.sin
        cos = np.cos
        L2 = 0.4 # the length of the thigh
        L3 = 0.4 # the length of the shin
        x,y,r,q1r,q2r,q1l,q2l, dx,dy,dr,dq1r,dq2r,dq1l,dq2l = self.state

        return np.array([ # this function is generated using matlab, I cannot find good interface from Pybullet
            [ 0, 0, - dq1r*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - dr*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - L3*dq2r*sin(q1r + q2r + r), - dq1r*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - dr*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - L3*dq2r*sin(q1r + q2r + r), - L3*dq1r*sin(q1r + q2r + r) - L3*dq2r*sin(q1r + q2r + r) - L3*dr*sin(q1r + q2r + r),                                                                                                                            0,                                                                                    0],
            [ 0, 0, - dq1r*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - dr*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - L3*dq2r*cos(q1r + q2r + r), - dq1r*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - dr*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - L3*dq2r*cos(q1r + q2r + r), - L3*dq1r*cos(q1r + q2r + r) - L3*dq2r*cos(q1r + q2r + r) - L3*dr*cos(q1r + q2r + r),                                                                                                                            0,                                                                                    0],
            [ 0, 0, - dq1l*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - dr*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - L3*dq2l*sin(q1l + q2l + r),                                                                                                                            0,                                                                                    0, - dq1l*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - dr*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - L3*dq2l*sin(q1l + q2l + r), - L3*dq1l*sin(q1l + q2l + r) - L3*dq2l*sin(q1l + q2l + r) - L3*dr*sin(q1l + q2l + r)],
            [ 0, 0, - dq1l*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - dr*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - L3*dq2l*cos(q1l + q2l + r),                                                                                                                            0,                                                                                    0, - dq1l*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - dr*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - L3*dq2l*cos(q1l + q2l + r), - L3*dq1l*cos(q1l + q2l + r) - L3*dq2l*cos(q1l + q2l + r) - L3*dr*cos(q1l + q2l + r)]
                       ])


    @property
    def dJ_gc(self):
        """
        The jacobian of the toes that are gc
        """
        return self.dJ_toe[(np.ones((2,2)).astype(bool) * self.gc_mask).T.reshape(-1), ...]


    @CTRL_COMPONENT
    def J_gc_bar(self):
        return self.RBD_A_inv @ self.J_gc.T @ self.gc_Lam


    @CTRL_COMPONENT
    def gc_N(self):
        """
        The null space of the operation space of gc
            gc_N = (I - J^T J_bar^T)
            where J_bar = A^{-1} J^T Lambda
        """
        return np.eye(GP.QDIMS) - self.J_gc.T @ self.J_gc_bar.T


    @CTRL_COMPONENT
    def DA(self):
        """
        This is the dynamics of the robot under Holonomic constraint of the ground
            dx = DA x + DB torque + Dg
            where x = [q,dq] 

        Note, the dynamics under Holonomic constraint means the solution of 
            | M    -J^T | | ddq |  =  | -RBD_B + torque |
            | J      O  | |Gamma|     | - dJdq          |
        """
        # First we get elements in M ddq = (I-J^T J_bar^T)(torque - RBD_B) - J^T Lambda dJdq
        return np.concatenate([
            np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)),np.eye(GP.QDIMS)], axis = 1),
            np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)), self.RBD_A_inv @ self.J_gc.T @ self.gc_Lam @ self.dJ_gc], axis = 1) # I am not sure to put the dJdq term in DA or DB
            # np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)),  np.zeros((GP.QDIMS,GP.QDIMS))], axis = 1) 
        ],axis = 0)


    @CTRL_COMPONENT
    def DB(self):
        return np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)),
                            self.RBD_A_inv @ self.gc_N @ np.diag([0,0,0,1,1,1,1])], axis = 0)


    @CTRL_COMPONENT
    def Dg(self):
        return np.concatenate([np.zeros(GP.QDIMS),
                - self.RBD_A_inv @ self.gc_N @ self.RBD_B] , axis = 0) # put J^T Lambda dJ dq here migh be the conventional way


    @staticmethod
    def setJointTorques(torque, mask = [1]*4):
        """
        arg torque: A four-element list 
        arg mask:   Whether to control this joint
        
        return: the setted torque of all the joints (dim7)
        """
        assert(len(torque) == 4)
        assert(len(mask) == 4)
        for i,ind in enumerate(qind[3:]):
            if(mask[i]):
                p.setJointMotorControl2(GP.robot,ind,p.POSITION_CONTROL,0,force=0)
                p.setJointMotorControl2(bodyIndex=GP.robot,
                                        jointIndex=ind,
                                        controlMode=p.TORQUE_CONTROL,
                                        force=max(min(torque[i], GP.MAXTORQUE[i+3]),-GP.MAXTORQUE[i+3]))
        return np.array([0]*3 + list(np.minimum(np.maximum(torque, -GP.MAXTORQUE[3:]), GP.MAXTORQUE[3:]) * np.array(mask)))


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


        ang = np.pi/15
        q_init = [0,0.8*np.cos(ang),0, np.pi - ang, 0, -(np.pi - ang), 0]
        CTRL.setState(q_init)
        
        for j in range (GP.QDIMS):
            force=GP.MAXTORQUE[j]
            pos=q_init[j]
            p.setJointMotorControl2(robot,qind(j),p.POSITION_CONTROL,pos,force=force)
        
        for i in range(10):
            p.stepSimulation()






robot=floor=numJoints=None
dt = 1e-3

physicsClient = p.connect(p.GUI) if (GP.GUIVIS) else p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
#p.setRealTimeSimulation(True)
p.setGravity(0, 0, GRAVITY)
p.setTimeStep(dt)

