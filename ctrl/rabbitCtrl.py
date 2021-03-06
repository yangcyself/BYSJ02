"""
The controller of a rabbit pybullet environment, providing functions like restart, set state, and run controller

The controllers in the 
"""

import globalParameters as GP
if GP.STARTSIMULATION:
    import pybullet as p
    import pybullet_data 
import os
import time
from globalParameters import qind
import numpy as np
import inspect



class CTRL_COMPONENT:
    def __init__(self,func):
        self.func = func
        self.name = self.func.__name__
        self.funcsig = [x.name for x in inspect.signature(self.func).parameters.values()]

    def __set__(self):
        raise ValueError("The control variable cannot be set implicitly, use method `set`")

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
        obj.ctrl_components[self.name]=self # use dict because the registor might want to remove the prvious regested component
        assert (all([k in self.funcsig for k in kwargs.keys()]) or not 
            "argument not declared in control component %s"%self.func.__name__)
        obj.ctrl_param[self.name] = kwargs


    def set(self, obj, value):
        """
            directly set the value instead of calculate it
        """
        obj.ctrl_value[self.name] = value
        obj.ctrl_flags[self.name] = True


    def resetFunc(self,obj,func):
        assert self.name == func.__name__ , "reset function should have the same name as the old one"
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

    def __init__(self, circulatePosition = True, auto_camera = True, *args,**kwargs):
        self.ctrl_components = {}
        self.ctrl_flags = {}  # whether control components has to refresh their computation
        self.ctrl_param = {}  # parameters of control components
        self.ctrl_value = {}  # results of control components
        self.ctrl_static = {} # static variables in control components
        self.t = 0
        self._JointTorqueCmd = np.zeros(4) # the command of joints torque. This value will be passed to joint each iteration
    
        self.resetFlags()
        self.callbacks = []

        #################################
        # regist some control components
        if(circulatePosition is not None and circulatePosition):
            if (type(circulatePosition) == tuple): 
                CTRL.circulatePosition.reg(self,limit = circulatePosition)
            else: 
                CTRL.circulatePosition.reg(self)
        if(auto_camera is not None and auto_camera):
            CTRL.auto_camera.reg(self)
        

    def _resetflag(self,k):
        self.ctrl_flags[k] = False


    def resetFlags(self):
        [self._resetflag(k) for k in self.ctrl_flags.keys()]
        self._JointTorqueCmd = np.zeros(4)


    def resetStatic(self):
        self.ctrl_static = {}

    def resetComponents(self):
        self.ctrl_components = {}
        self.ctrl_param = {}


    def _setJointTorques(self):
        """
        ! This method is only to be called in `step`. The interface for set torque is `setJointTorques` 
            This implementation is to make sure that `p.setJointMotorControl2` is called only once in each step. 
            `p.setJointMotorControl2`'s command will be plused instead of overwritten if called more than once before one step
        """
        p.setJointMotorControlArray(GP.robot,qind[3:],p.POSITION_CONTROL,[0]*4,forces=[0]*4)
        p.setJointMotorControlArray(bodyIndex=GP.robot,
                                jointIndices=qind[3:],
                                controlMode=p.TORQUE_CONTROL,
                                forces=self._JointTorqueCmd)


    def step(self,T = None,sleep=None):
        assert not GP.KEYBOARD_INTERRPUT,"Interrupted"
        T = dt if T is None else T
        for i in range(int(T/dt+0.5)):
            # Call all regested control components
            try:
                self.resetFlags()
                [c(self) for c in self.ctrl_components.values()]    
                callres = [c(self) for c in self.callbacks]
                if(any(callres)): # use the call backs as break point checkers
                    return callres

                self._setJointTorques()
                p.stepSimulation()
                self.t += dt
                if(sleep is not None):
                    time.sleep(sleep)
                else:
                    time.sleep(dt)
            except KeyboardInterrupt as ex:
                GP.KEYBOARD_INTERRPUT = not GP.CATCH_KEYBOARD_INTERRPUT
                return [c(self) for c in self.callbacks]
            except p.error as ex:
                if("Not connected to physics server" in str(ex)):
                    return None
                raise ex
        return None

    @CTRL_COMPONENT
    def state(self):
        # return the current state in the format of [q, dq]
        x =  np.array([s[:2] for s in p.getJointStates(GP.robot,list(qind))]).T.reshape(-1)
        # normalize the state making the q1_r,q1_l be in the range of 0~2pi, r, q2_r q2_l be in the range -pi~pi
        x[[2, 4, 6]] = x[[2, 4, 6]] - ((x[[2, 4, 6]] + np.math.pi)//(2*np.math.pi))*(2*np.math.pi)
        x[[3, 5]] = x[[3, 5]] % (2*np.math.pi)
        return x

    @CTRL_COMPONENT
    def CoMs(self):
        # return the Center of Mass of the links in the global frame
        #   [body, right_thigh, right_shin, right_toe, left_thign, left_shin, left_toe], body is ind 2
        linkInd = np.arange(2,9)
        return np.array([np.array(s[0])[GP.PRISMA_AXIS] for s in p.getLinkStates(GP.robot,linkInd)])


    @CTRL_COMPONENT
    def J_toe(self):
        J = [p.calculateJacobian(GP.robot,i,
            [0,0,0], # the point on the specified link
            list(self.state[:7]),
            [0 for a in range(7)], # joint velocities
            [0 for a in range(7)]) # desired joint accelerations
        for i in [5,8]] # calculates the linear jacobian and the angular jacobian
        # Each element of J is tuple (J_x, J_r), the jacobian of transition and rotation
        return np.concatenate([np.array(j[0])[GP.PRISMA_AXIS] for j in J], axis = 0)


    @CTRL_COMPONENT
    def contact_mask(self):
        """
        A boolean tuple representing whether the foot is interacting with the ground
        """
        contact_mask = np.array([len(p.getContactPoints(GP.robot,GP.floor,i)) for i in [5,8]]).astype(np.bool)
        return contact_mask

    @CTRL_COMPONENT
    def gc_mask(self,constant = None,**kwargs):
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
            RBD_A ddq + RBD_B = torque + J.T F
        """
        return np.array(p.calculateInverseDynamics(robot, list(self.state[:GP.QDIMS]), list(self.state[GP.QDIMS:]), [0.0] * GP.QDIMS))


    @CTRL_COMPONENT
    def gc_Lam(self):
        """
        This is the lambda of the operational space of the gc
            Lambda = (J A^{-1} J^T)^-1
        """
        lam = np.linalg.inv(self.J_gc @ self.RBD_A_inv @ self.J_gc.T)
        # assert(np.isclose(lam,lam.T).all()) 
        return lam


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
        return np.concatenate([
            np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)),np.eye(GP.QDIMS)], axis = 1),
            np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)), - self.RBD_A_inv @ self.J_gc.T @ self.gc_Lam @ self.dJ_gc], axis = 1) # I am not sure to put the dJdq term in DA or DB
            # np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)),  np.zeros((GP.QDIMS,GP.QDIMS))], axis = 1) 
        ],axis = 0)


    @CTRL_COMPONENT
    def DB(self):
        return np.concatenate([np.zeros((GP.QDIMS,GP.QDIMS)),
                            self.RBD_A_inv @ self.gc_N @ np.diag([0,0,0,1,1,1,1])], axis = 0)


    @CTRL_COMPONENT
    def Dg(self):
        return np.concatenate([np.zeros(GP.QDIMS),
                # - self.RBD_A_inv @ self.gc_N @ self.RBD_B - self.RBD_A_inv @ self.J_gc.T @ self.gc_Lam @ self.dJ_gc @ self.state[GP.QDIMS:]] , axis = 0) # put J^T Lambda dJ dq here migh be the conventional way
                - self.RBD_A_inv @ self.gc_N @ self.RBD_B] , axis = 0) 


    @CTRL_COMPONENT
    def FrAB(self):
        """
        (Fr_A, Fr_b): The reaction force Fr in holonomic constraint given u is: Fr = Fr_A @ u + Fr_b

            The calculation is from the reaction force in gc operation space:
            Fr = - Jc_bar (- g + B u) - Lambda dJ_gc dq  
        """
        FrA = - self.J_gc_bar.T @ np.diag([0,0,0,1,1,1,1])
        Frb = self.J_gc_bar.T @ self.RBD_B - self.gc_Lam @ self.dJ_gc @ self.state[GP.QDIMS:]
        return (FrA, Frb)


    def setJointTorques(self, torque, mask = [1]*4):
        """
        arg torque: A four-element list 
        arg mask:   Whether to control this joint
        
        return: the setted torque of all the joints (dim7)
        """
        assert(len(torque) == 4)
        assert(len(mask) == 4)
        mask = np.array(mask).astype(bool)
        self._JointTorqueCmd[mask] = np.minimum(np.maximum(torque, -GP.MAXTORQUE[3:]), GP.MAXTORQUE[3:])[mask]
        return np.array([0]*3 + list(self._JointTorqueCmd))


    @CTRL_COMPONENT
    def circulatePosition(self,limit = (-9,9)):
        """
        keep the robot position in the limitation by transpolation
        """
        if (self.state[0] < limit[0] or self.state[0] > limit[1]):
            s = self.state
            s[0] = limit[0] + (s[0]-limit[0])%(limit[1]-limit[0])
            self.setState(s)
            return True
        return False
    

    @CTRL_COMPONENT
    def auto_camera(self,cameraDistance = 3, targetheight = 0.5, pitch = -20):
        target = [0,0,targetheight]
        target[GP.PRISMA_AXIS[0]] = self.state[0]
        p.resetDebugVisualizerCamera(cameraDistance,0,pitch,target)
        return True


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
        p.setGravity(0,0,GP.GRAVITY)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor = p.loadURDF("plane.urdf")

        # robotStartPos = [0, 0, -2+0.63]
        robotStartPos = [0, 0, 0]
        robotStartOrientation = p.getQuaternionFromEuler([0., 0, 0])
        robot = p.loadURDF(GP.URDF_FILE, robotStartPos,robotStartOrientation)

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



dt = GP.DT
if GP.STARTSIMULATION:

    robot=floor=numJoints=None

    physicsClient = p.connect(p.GUI) if (GP.GUIVIS) else p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #p.setRealTimeSimulation(True)
    p.setGravity(0, 0, GP.GRAVITY)
    p.setTimeStep(dt)