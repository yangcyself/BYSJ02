from ctrl.rabbitCtrl import *


class playBackCTRL(CTRL):
    """
        The controller that replay a trajectory of input
    """


    def __init__(self,traj):
        super().__init__()
        self.traj = traj
        playBackCTRL.playback.reg(self)
    
    def setStateT(self,t = 0):
        t = np.argmin(abs(np.array(self.traj["t"])-t))
        self.setState(self.traj["state"][t])
        
    @CTRL_COMPONENT
    def playback(self):
        t = np.argmin(abs(np.array(self.traj["t"])-self.t))
        return self.setJointTorques(self.traj["u"][t][3:])

