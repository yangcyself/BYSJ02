from ctrl.rabbitCtrl import *


class playBackCTRL(CTRL):
    """
        The controller that replay a trajectory of input
    """


    def __init__(self,traj,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.traj = traj
        playBackCTRL.playback.reg(self)
        self.trajLength = traj["t"][-1]
    
    def setStateT(self,t = 0):
        self.resetFlags()
        t = np.argmin(abs(np.array(self.traj["t"])-t))
        self.setState(self.traj["state"][t])

    @CTRL_COMPONENT
    def playback(self):
        t = np.argmin(abs(np.array(self.traj["t"])-self.t))
        return self.setJointTorques(self.traj["u"][t][3:])

    def playstate(self,*trange, dt = GP.DT,sleep = GP.DT):
        for t in np.arange(*trange,step = GP.DT):
            self.setStateT(t)
            time.sleep(sleep)

    def getTraj(self,t):
        t = np.argmin(abs(np.array(self.traj["t"])-t))
        return {k:v[t] if type(v)==list and len(v)>t else None for k,v in self.traj.items()}


    def numericalJac(self, func,du = 1e-3):
        """
        numerically compute the jacobian of a value with regard to u (from the current state)
        func: lambda obj: obj.ctrlcomponent
        """
        pass

