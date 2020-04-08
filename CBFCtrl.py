"""
The controller whose lowest part is a CBF(-CLF)-QP
"""
from rabbitCtrl import *
import autograd.numpy as np

class CBF_CTRL(CTRL):

    def __init__(self):
        super().__init__()
        self.HA = None
        self.Hb = None
        self.Hc = None
    

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
            dHx = DHA @ Hx + DHB + u
        """
        return self.DA


    @property
    def DHB(self):
        """
        The dyanmic transition matrix in the space of Hx: (currently they are the set the same)
            dHx = DHA @ Hx + DHB + u
        """
        return self.DB


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
        dB_c = 2 * self.Hx.T @ self.HA @self.DHA @ self.Hx
        return (dB_b, dB_c)


    @CTRL_COMPONENT
    def CBF_CLF_QP(self):
        """
        (self.CBF, self.CLF) -> setJointTorques()
        """
        pass    
