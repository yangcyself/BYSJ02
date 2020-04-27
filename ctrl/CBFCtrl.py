"""
The controller whose lowest part is a CBF(-CLF)-QP
"""
from ctrl.rabbitCtrl import *
from scipy.optimize import minimize


class CBF_CTRL(CTRL):

    def __init__(self):
        super().__init__()
        self.HA = []
        self.Hb = []
        self.Hc = []
        self.Hw = []
        self.HisCBF = [] # Ture for CBF and False for CLF
        self._v = False  # a public verbose flag, for convinence
        CBF_CTRL.CBF_CLF_QP.reg(self)

    def addCBF(self, A,b,c, w = 1):
        self.HA.append(A)
        self.Hb.append(b)
        self.Hc.append(c)
        self.Hw.append(w)
        self.HisCBF.append(True)
    
    def addCLF(self, A,b,c, w = 1):
        self.HA.append(A)
        self.Hb.append(b)
        self.Hc.append(c)
        self.Hw.append(w)
        self.HisCBF.append(False)
    

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
    def HF(self): 
        """
        (self.HA, self.Hb, self.Hc, self.Hx) -> h(x)
        use a list comprehension to calculate the each function (can be CBF or CLF)
        """
        return [self.Hx.T @ A @ self.Hx + b @ self.Hx + c
                for A,b,c in zip(self.HA,self.Hb,self.Hc)]


    @CTRL_COMPONENT
    def dHF(self): 
        """
        The vector and constant that forms the time derivative of BF 
            dBF = dB_b @ u + dB_c
        (self.HA, self.Hb, self.Hc, self.Hx, self.DHA, self.DHB) -> (dB_b, dB_c)
        use a list comprehension to calculate the each function (can be CBF or CLF)
        """
        # dB_b = (2 * self.Hx.T @ self.HA + self.Hb.T) @ self.DHB  
        # dB_c = (2 * self.Hx.T @ self.HA + self.Hb.T)@(self.DHA @ self.Hx + self.DHg)
        return [( (2 * self.Hx.T @ A + b) @ self.DHB  ,# dB_b,
                (2 * self.Hx.T @ A + b)@(self.DHA @ self.Hx + self.DHg)) 
                for A,b,c in zip(self.HA,self.Hb,self.Hc)]


    @CTRL_COMPONENT
    def CBF_CLF_QP(self, mc_b = 1, ml = 0.00001):
        """
        ((self.BF,self.dBF), (self.CF, self.dCF), self.HisCBF) -> setJointTorques()
        args mc: gamma
        args ml: lambda weight for normalization term in obj_func
        """


        def obj(u):
            return ml * u.T@u + sum([ max(0, w * (dL_b @ u + dL_c + LF))
                                for (LF,(dL_b, dL_c),isB, w) in zip(self.HF,self.dHF,self.HisCBF,self.Hw) if not isB])
        def obj_jac(u):
            return ml * 2*u + sum([  w * dL_b * (dL_b @ u + dL_c + LF > 0)
                                for (LF,(dL_b, dL_c),isB, w) in zip(self.HF,self.dHF,self.HisCBF,self.Hw) if not isB])

    
        def CBF_cons_gen(dB_b,dB_c,BF):
            return lambda u: mc_b * (dB_b @ u +  dB_c) + BF
        def CBF_cons_jac_gen(dB_b,dB_c):
            return lambda u: mc_b * dB_b

        constraints = [{'type':'ineq','fun': CBF_cons_gen(dB_b,dB_c,BF), "jac":CBF_cons_jac_gen(dB_b,dB_c) }
                        for (BF,(dB_b, dB_c),isB, w) in zip(self.HF,self.dHF,self.HisCBF,self.Hw) if isB]

        x0 = np.random.random(7)
        # x0 = np.ones(2)

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
        if(self._v):
            print("Torque:", res_x)
            print("Fr:", self.J_gc_bar.T @ res_x)
        return self.setJointTorques(res_x[3:])
    

