"""
The combination of CBF and IOlinearization controller. 
    The linearization forms the "CLF", and some simple CBF condition is set in the CBF-QP
"""

import sys
sys.path.append(".")

from ctrl.IOLinearCtrl import *
from ctrl.CBFCtrl import *

class CBF_WALKER_CTRL(CBF_CTRL, IOLinearCTRL):
    """
        Add two state representing the angel from the stance toe and the swing toe to the torso
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        CBF_WALKER_CTRL.CBF_CLF_QP.reg(self) # This operation will cover the `CBF_CLF_QP` in parent class
        self.LOG_CBF_SUCCESS = True # flag representing the status of CBF
        self.LOG_ResultFr = None
        self.LOG_CBF_ConsValue = None
        self.LOG_CBF_Value = None
        self.LOG_CBF_Drift = None


    @CTRL_COMPONENT
    def CBF_Ftr_Mat(self):
        """
        (Map, reverse_Map)
            Return the Matrix that map the state of the system to the feature of the CBF, the reverse_Map maps the Ftr to CBF's state
                The angle is r + q_1 + 0.5 * q_2

            NOTE: For current implementation, the Feature also uses state[0] as the theta of swing leg
                See `VC_c` in `IOLinearCtrl` for detail
            
            Three state are added to the seven. The first is the about the stance leg and the second is about the swing leg, the third is the Tau
        
        """
        redStance, theta_plus, Labelmap = self.STEP_label
        leg_angle_mat = np.array([1,0.5]) # the q_1 + 0.5 q_2
        Mat = np.concatenate([
                np.concatenate([np.eye(7),np.zeros((6+7,7))],axis = 0),
                np.concatenate([np.zeros((7+3,7)), np.eye(7), np.zeros((3,7))],axis = 0)
        ],axis = 1)

        rMat = Mat.copy().T

        Mat[[7,8],2] = Mat[[17,18],9] = 1

        Mat[7,3:7] = Labelmap[:,:2] @ leg_angle_mat # stance toe
        Mat[8,3:7] = Labelmap[:,2:] @ leg_angle_mat # swing toe

        Mat[17,10:14] = Labelmap[:,:2] @ leg_angle_mat # stance toe
        Mat[18,10:14] = Labelmap[:,2:] @ leg_angle_mat # swing toe
        
        Mat[9,0] = Mat[19,0] = 1/self.stepLength # Tau

        return Mat, rMat


    @property
    def Hx(self):
        """
        The "state" note that this is not q, the CBF is a quadratic form of q (currently they are setted the same)
        self.state -> self.Hx 
        """
        Mat, rMat = self.CBF_Ftr_Mat
        x = Mat @ self.state
        # add the thetaplus
        redStance, theta_plus, Labelmap = self.STEP_label
        x[9] -= theta_plus / self.stepLength
        return x


    @property
    def DHA(self):
        """
        The dyanmic transition matrix in the space of Hx: (currently they are the set the same)
            dHx = DHA @ Hx + DHB @ u + DHg
        """
        Mat, rMat = self.CBF_Ftr_Mat
        return  Mat @ self.DA @ rMat


    @property
    def DHB(self):
        """
        The dyanmic transition matrix in the space of Hx: (currently they are the set the same)
            dHx = DHA @ Hx + DHB @ u + DHg
        """
        Mat, rMat = self.CBF_Ftr_Mat
        return Mat @ self.DB


    @property
    def DHg(self):
        """
        The dyanmic transition matrix in the space of Hx: (currently they are the set the same)
            dHx = DHA @ Hx + DHB @ u + DHg
        """
        Mat, rMat = self.CBF_Ftr_Mat
        return Mat @ self.Dg


    @CTRL_COMPONENT
    def CBF_CLF_QP(self, mc_b = 1, ml = 0.00001):
        """
        ((self.BF,self.dBF), (self.CF, self.dCF), self.HisCBF) -> setJointTorques()
        args mc: gamma
        args ml: lambda weight for normalization term in obj_func
        """

        IO_u = self.IOLin # this operation sets JointTorques commands, but the command will be flushed later

        def obj(u):
            return ml * (u-IO_u).T@(u-IO_u) + sum([ max(0, w * (dL_b @ u + dL_c + LF))
                                for (LF,(dL_b, dL_c),isB, w) in zip(self.HF,self.dHF,self.HisCBF,self.Hw) if not isB])
        def obj_jac(u):
            return ml * 2*(u-IO_u) + sum([  w * dL_b * (dL_b @ u + dL_c + LF > 0)
                                for (LF,(dL_b, dL_c),isB, w) in zip(self.HF,self.dHF,self.HisCBF,self.Hw) if not isB])

        def CBF_cons_gen(dB_b,dB_c,BF):
            return lambda u: mc_b * (dB_b @ u +  dB_c) + BF
        def CBF_cons_jac_gen(dB_b,dB_c):
            return lambda u: mc_b * dB_b

        CBF_CONS = [CBF_cons_gen(dB_b,dB_c,BF) for (BF,(dB_b, dB_c),isB, w) in zip(self.HF,self.dHF,self.HisCBF,self.Hw) if isB]
        constraints = [{'type':'ineq','fun': CBF_cons_gen(dB_b,dB_c,BF), "jac":CBF_cons_jac_gen(dB_b,dB_c) }
                        for (BF,(dB_b, dB_c),isB, w) in zip(self.HF,self.dHF,self.HisCBF,self.Hw) if isB]

        if(sum(self.gc_mask)>0):
            FrA, Frb = self.FrAB
            constraints += [{'type':'ineq','fun': lambda u: (FrA @ u + Frb)[1], "jac": lambda u: FrA[1,:] }]
            if(sum(self.gc_mask)==1):
                constraints += [{'type':'ineq','fun': lambda u: np.array([[-1,GP.MU],[1,GP.MU]]) @ (FrA @ u + Frb), 
                                            "jac": lambda u: np.array([[-1,GP.MU],[1,GP.MU]]) @ FrA }
                                            for i in range(0,FrA.shape[0],2)]

        x0 = np.random.random(7)

        bounds = np.concatenate([-GP.MAXTORQUE[:,None],GP.MAXTORQUE[:,None]],axis = 1)

        options = {"maxiter" : 500, "disp"    : self._v}
        res = minimize(obj, x0, bounds=bounds,options = options,jac=obj_jac,
                    constraints=constraints, method =  'SLSQP') # 'trust-constr' , "SLSQP"
        # assert(cbf(res.x)>-1e-9)
        res_x = np.nan_to_num(res.x,0)
        # if(not CBF_cons(res_x)>-1e-9):
        #     # badpoints.append(state)
        #     print("bad CBF: ", CBF_cons(res_x))
        #     print(CBF_cons(res_x))
        
        # Use a flag to represent the state of optimization
        self.LOG_CBF_SUCCESS = res.success
        self.LOG_ResultFr = FrA @ res_x + Frb if(sum(self.gc_mask)>0) else np.zeros(2)
        if not self.LOG_CBF_SUCCESS:
            print("Optimization Message:", res.message)
        # print("CBF Result: ",[c(res_x) for c in CBF_CONS])
        self.LOG_CBF_ConsValue = [c(res_x) for c in CBF_CONS]
        self.LOG_CBF_Value = self.HF
        self.LOG_CBF_Drift = [dB_c for dB_b, dB_c in  self.dHF]
        if(self._v):
            print("Torque:", res_x)
            print("Fr:", self.J_gc_bar.T @ res_x)
        return self.setJointTorques(res_x[3:])
    