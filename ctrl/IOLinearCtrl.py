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
        self.stepLength = 0.0779 -( -0.0797)
        IOLinearCTRL.IOLin.reg(self)


    @CTRL_COMPONENT
    def STEP_label(self, firstFlag = True):
        """
        Maintain the left step and right step
        (bool, double, double[4x4])
            First argument is whether q1 q2 (the red leg) is the stance leg
            the second is theta_plus. The theta_variable of the init of this step
            The relabel matrix (4x4) to map the command of IOcmd and swing legs

        Forms a four state FSM
            (redStance: True, lifted = False) -> (redStance: True, lifted = True) v
           ^  (redStance: False, lifted = True) <- (redStance: False, lifted = False)
        """
        # set static variable redStance if it has not been declared
        redStance = self.ctrl_static["redStance"] = self.ctrl_static.get("redStance",firstFlag)
        lifted = self.ctrl_static["lifted"] = self.ctrl_static.get("lifted",False)
        theta_plus = self.ctrl_static["theta_plus"] = self.ctrl_static.get("theta_plus",self.state[0] - 0.001)

        # check swing leg lifted
        if lifted:
            # flip flag when the swing leg has contacted with the ground
            if(self.contact_mask[int(redStance)]):
                redStance = self.ctrl_static["redStance"] = not redStance
                lifted = self.ctrl_static["lifted"] = False
                theta_plus = self.ctrl_static["theta_plus"] = self.state[0] - 0.001

        if not lifted:  # this means lifted can be flipped immediatly after being set to false 
            # Judge whether swing foot has lifted
            if(not self.contact_mask[int(redStance)]):
                lifted = self.ctrl_static["lifted"] = True

        Labelmap = np.eye(4)
        if(not redStance):
            Labelmap = np.concatenate([Labelmap[:,2:], Labelmap[:,:2]], axis = 1)
        
        return redStance, theta_plus, Labelmap


    @CTRL_COMPONENT
    def VC_c(self):
        """
        (tau, s, ds, phase_len)
            s: The clock variable of the virtual constraint from one to zero
            Phase_len: The total time for the phase to go from 0 to 1, used to calculate the dt and ddt
                Here I use the x-axis of the robot
        """
        redStance, theta_plus, Labelmap = self.STEP_label
        tau = (self.state[0] - theta_plus) / self.stepLength
        return tau, self.state[0], self.state[7], self.stepLength
    

    @CTRL_COMPONENT
    def IOcmd(self, Balpha = None):
        """
            (Y_c, dY_c, ddY_c) where each cmd is a length-4 vector
            Here I use a 5-order bzser function to represend the command 
        """
        # phi(k) + alpha(k,m+1)*factorial(M)/(factorial(m)*factorial(M-m))*s^m*(1-s)^(M-m);
        K,M = Balpha.shape # K: The number of the virtual constraint, M: the order
        M = M-1
        tau, s, ds, phase_len = self.VC_c
        phi = np.sum(
            [tau**m * (1-tau)**(M-m) * 
                Balpha[:,m] * factorial(M) / (factorial(m)*factorial(M-m)) for m in range(M+1)], 
            axis = 0)

        dphidc = np.sum(
            [(m * tau**(m-1) * (1-tau)**(M-m) - (M-m)*(1-tau)**(M-m-1) * tau**m ) * 
                Balpha[:,m] * factorial(M) / (factorial(m)*factorial(M-m)) for m in range(M+1)], 
            axis = 0) / phase_len

        ddphiddc = np.sum(
            [(m * (m-1) * tau**(m-2) * (1-tau)**(M-m) - 2*(M-m) * m * tau**(m-1) * (1-tau)**(M-m-1) + (M-m) * (M-m-1) * tau**m * (1-tau)**(M-m-2))* 
                Balpha[:,m] * factorial(M) / (factorial(m)*factorial(M-m)) for m in range(M+1)], 
            axis = 0) / (phase_len ** 2)

        redStance, theta_plus, Labelmap = self.STEP_label
        return Labelmap@ phi, Labelmap@ dphidc * ds, Labelmap@ ddphiddc * ds**2 *0 ### NOTE disable the feedforward of the acc of the gait


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
        except np.linalg.LinAlgError as ex:
            if("Singular matrix" not in str(ex)):
                raise(ex)
            u = np.linalg.pinv(self.VC_H @ (self.DA @ DB)[:GP.QDIMS,:]) @ (ddYC + fw)
        return self.setJointTorques(u)


