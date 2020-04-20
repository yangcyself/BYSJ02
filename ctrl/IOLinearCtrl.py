"""
The IO linearlization controller

This controller is a basic one for walking gaits
"""

from ctrl.rabbitCtrl import *
from math import factorial

class IOLinearCTRL(CTRL):
    """
        The IO linearlization controller
        The control strategy is using virtual constraint
            The constraint is: (Note that H is only constraint about position)
            Y = H x
            dY = H dx
            ddY = H DA (DA x + DB u + dg)

            And we have the control rule
            u = (H DA DB)^{-1} ( ddYC - H DA (DA x + dg) )
            where ddYC is  kp * (Y_c-Y) + kd * (dY_c - dY) + ddY_c

            And the Y_c, dY_c, ddY_c are commands calculated by IOcmd.
    """

    def __init__(self):
        super().__init__()
        self.VC_H = np.concatenate([np.zeros((4,3)), np.eye(4)], axis=1)
        self.step_t = 1
        IOLinearCTRL.IOLin.reg(self)


    @CTRL_COMPONENT
    def VC_c(self):
        """
        (s, phase_len)
            s: The clock variable of the virtual constraint from one to zero
            Phase_len: The total time for the phase to go from 0 to 1, used to calculate the dt and ddt
                Here I use the x-axis of the robot
        """
        return self.state[0], self.step_t
    

    @CTRL_COMPONENT
    def IOcmd(self, Balpha = None):
        """
            (Y_c, dY_c, ddY_c) where each cmd is a length-4 vector
            Here I use a 5-order bzser function to represend the command 
        """
        # phi(k) + alpha(k,m+1)*factorial(M)/(factorial(m)*factorial(M-m))*s^m*(1-s)^(M-m);
        K,M = Balpha.shape # K: The number of the virtual constraint, M: the order
        s, phase_len = self.VC_c
        phi = np.sum(
            [s**m * (1-s)**(M-m) * 
                Balpha[:,m] * factorial(M) / (factorial(m)*factorial(M-m)) for m in range(M)], 
            axis = 0)

        dphidc = np.sum(
            [m * s**(m-1) * (1-s)**(M-m) * 
                Balpha[:,m] * factorial(M) / (factorial(m)*factorial(M-m)) for m in range(M)], 
            axis = 0) / phase_len

        ddphiddc = np.sum(
            [(m * (m-1) * s**(m-2) * (1-s)**(M-m) - 2*(M-m) * m * s**(m-1) * (1-s)**(M-m-1) + (M-m) * (M-m-1) * s**m * (1-s)**(M-m-2))* 
                Balpha[:,m] * factorial(M) / (factorial(m)*factorial(M-m)) for m in range(M)], 
            axis = 0) / (phase_len ** 2)
        
        return phi, dphidc, ddphiddc


    @CTRL_COMPONENT
    def IOLin(self, kp = 100, kd = 20):
        """
        ddYC  =  kp * (Y_c-Y) + kd * (dY_c - dY) + ddY_c
        """
        Y_c, dY_c, ddY_c = self.IOcmd
        Y = self.VC_H @ self.state[:GP.QDIMS]
        dY = self.VC_H @ self.state[GP.QDIMS:]

        ddYC =  kp * (Y_c - Y) + kd * (dY_c - dY) + ddY_c
        fw = -self.VC_H @ (self.DA @(self.DA @self.state + self.Dg))[:GP.QDIMS]
        
        DB = self.DB[:,3:]
        try:
            u = np.linalg.inv(self.VC_H @ (self.DA @ DB)[:GP.QDIMS,:]) @ (ddYC + fw)
        except np.LinAlgError as ex:
            if("Singular matrix" not in str(ex)):
                raise(ex)
            u = np.linalg.pinv(self.VC_H @ (self.DA @ DB)[:GP.QDIMS,:]) @ (ddYC + fw)
        return self.setJointTorques(u)


