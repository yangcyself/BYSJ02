"""
Some utilities for generating better samples
"""
import numpy as np

class GaussionProcess:
    """
    This class maintains the states of gaussion process
        with the mu and sigma as described in eq 1 and eq 2 in the paper
    
    arg: sigma2 The prior std^2 of the observation
    """
    def __init__(self, kernel = None, sigma2 = 0):
        self.kernel = (lambda x,y: np.exp(-np.linalg.norm(x-y))) if kernel is None else kernel
        self.sigma2 = sigma2
        self.K = np.array([[]]) # the kernel matrix of already observed points
        self.kfunlist = []
        self.k = lambda x:np.array([[f(x)] for f in self.kfunlist]) # the function that returns the vector of a point ot observed points
        self.y = []


    def __call__(self,x):
        # return the posterior mu and sigma inferred from current gaussian process
        ktx = self.k(x)
        mu = ktx.T @ np.linalg.inv(self.K+self.sigma2*np.eye(len(self.y))) @ np.array(self.y)
        sigma = self.kernel(x,x) - ktx.T @ np.linalg.inv(self.K+self.sigma2*np.eye(len(self.y))) @ ktx
        return mu, np.sqrt(sigma)
    

    def addObs(self,x,y): 
        """
        add an observation, x is the state, y is the value
        """
        # extend the K matrix
        sigma12 = self.k(x)
        kxx = np.array(self.kernel(x,x))
        if(not min(self.K.shape)):
            self.K = kxx.reshape((1,1))
        else:
            self.K = np.concatenate([
                        np.concatenate([self.K,     sigma12],axis = 1),
                        np.concatenate([sigma12.T,  kxx.reshape((1,1))],axis = 1),
                ],axis = 0)

        self.kfunlist.append(lambda x_: self.kernel(x,x_))
        self.y.append(y)


def randomSample(trajdict, tauInd, num=20, IncludeCollisionPoints = 0, period = None, IncludeInit = True):
    """
    randomly sample `num` states from the trajdict["Hx"], if IncludeAllCollisionPoints, then IncludeAllCollisionPoints
    
    return an array of points [num x dim]
    """
    if(period is None):
        startInd,endInd = 0,len(trajdict["t"])
    elif(type(period)==tuple):
        startInd = np.argmin(abs(np.array(trajdict["t"])-period[0]))
        endInd = np.argmin(abs(np.array(trajdict["t"])-period[1]))
    else:
        startInd = 0
        endInd = np.argmin(abs(np.array(trajdict["t"])-period))
    Hx = np.array(trajdict["Hx"])[startInd:endInd]
    tau = Hx[:,tauInd]
    collideInd = np.where(abs(tau[1:]-tau[:1])>0.1)[0]
    CollisonPoints = Hx[np.random.choice(collideInd,size = IncludeCollisionPoints)]
    ChoosePoints = Hx[np.random.choice(len(Hx),size = num)]
    if(IncludeInit):
        ChoosePoints = list(ChoosePoints) + [Hx[0]]
    return np.array(list(CollisonPoints) + list(ChoosePoints))
    