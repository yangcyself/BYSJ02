"""
The controller whose lowest part is a CBF(-CLF)-QP
"""
from rabbitCtrl import *
import autograd.numpy as np
from scipy.optimize import minimize


class CBF_CTRL(CTRL):

    def __init__(self):
        super().__init__()
        self.HA = None
        self.Hb = None
        self.Hc = None
        self._v = False  # a public verbose flag, for convinence
        CBF_CTRL.CBF_CLF_QP.reg(self)

    @property # use property because I don't what to cache them
    def Hx(self):
        """
        The "state" note that this is not q, the CBF is a quadratic form of q (currently they are setted the same)
        self.state -> self.Hx 
        """
        return self.state


    @property
    def DHA(self):
        """
        The dyanmic transition matrix in the space of Hx: (currently they are the set the same)
            dHx = DHA @ Hx + DHB @ u + DHg
        """
        return self.DA


    @property
    def DHB(self):
        """
        The dyanmic transition matrix in the space of Hx: (currently they are the set the same)
            dHx = DHA @ Hx + DHB @ u + DHg
        """
        return self.DB


    @property
    def DHg(self):
        """
        The dyanmic transition matrix in the space of Hx: (currently they are the set the same)
            dHx = DHA @ Hx + DHB @ u + DHg
        """
        return self.Dg


    @CTRL_COMPONENT
    def BF(self): 
        """
        (self.HA, self.Hb, self.Hc, self.Hx) -> h(x)
        """
        return self.Hx.T @ self.HA @ self.Hx + self.Hb @ self.Hx + self.Hc


    @CTRL_COMPONENT
    def dBF(self): 
        """
        The vector and constant that forms the time derivative of BF 
            dBF = dB_b @ u + dB_c
        (self.HA, self.Hb, self.Hc, self.Hx, self.DHA, self.DHB) -> (dB_b, dB_c)
        """
        dB_b = (2 * self.Hx.T @ self.HA + self.Hb.T) @ self.DHB  
        dB_c = (2 * self.Hx.T @ self.HA + self.Hb.T)@(self.DHA @ self.Hx + self.DHg)
        return (dB_b, dB_c)


    @CTRL_COMPONENT
    def CBF_CLF_QP(self, mc=1):
        """
        ((self.BF,self.dBF), self.CLF) -> setJointTorques()
        args mc: gamma
        """

        def obj(u):
            return u.T@u
        def obj_jac(u):
            return 2*u

        (dB_b, dB_c) = self.dBF

        def CBF_cons(u):
            return mc * (dB_b @ u +  dB_c) + self.BF
        def CBF_cons_jac(u):
            return mc * dB_b

        constraints = {'type':'ineq','fun': CBF_cons, "jac":CBF_cons_jac }

        x0 = np.random.random(7)
        # x0 = np.ones(2)

        bounds = np.concatenate([-GP.MAXTORQUE[:,None],GP.MAXTORQUE[:,None]],axis = 1)


        options = {"maxiter" : 500, "disp"    : self._v}
        res = minimize(obj, x0, bounds=bounds,options = options,jac=obj_jac,
                    constraints=constraints, method =  'SLSQP') # 'trust-constr' , "SLSQP"
        # assert(cbf(res.x)>-1e-9)
        res_x = np.nan_to_num(res.x,0)
        if(not CBF_cons(res_x)>-1e-9):
            # badpoints.append(state)
            print("bad CBF: ", CBF_cons(res_x))
            print(CBF_cons(res_x))
        if(self._v):
            print("Torque:", res_x)
            print("Fr:", self.J_gc_bar.T @ res_x)
        return self.setJointTorques(res_x[3:])
    

if __name__ == "__main__":
    CBF_CTRL.restart()
    ct = CBF_CTRL()
    # The hand designed CBF for torso position
    mc = 1
    ct.HA = np.zeros((14,14))
    ct.HA[1,1] = mc
    ct.HA[1,8] = ct.HA[8,1] = 1

    ct.Hb = np.zeros(14)
    ct.Hc = -mc * 0.5 * 0.5

    ct.step(10)