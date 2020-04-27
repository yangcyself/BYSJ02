"""
The controller whose lowest part is a CBF(-CLF)-QP
"""
from ctrl.CBFCtrl import *
from scipy.optimize import minimize


class SteppingStoneCTRL(CBF_CTRL):

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

