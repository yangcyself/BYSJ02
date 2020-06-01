import numpy as np

def getMeanPeriod(traj,tauInd):
    """
    Input a traj dict
    """
    tau = np.array(traj["Hx"])[:,tauInd]
    collideInd = np.where(abs(tau[1:]-tau[:1])>0.1)[0]
    print("len(collideInd) :",len(collideInd))
    
    collideTimes = np.array(traj["t"])[collideInd]
    return np.mean(collideTimes[1:] - collideTimes[:-1])