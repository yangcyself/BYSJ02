import numpy as np
import matplotlib.pyplot as plt

def QuadContour(A,B,C, X, Xbase = 0, Ybase = 1,x0 = None):
    """
    return a list of points (x,y) where X is from the set list X

    args: Xbase, Ybase are the axis to be plotted, We assume the value of all other spaces are 0
    """
    def vec_u(u):
        if(type(u)==int):
            res = np.zeros((A.shape[0]))
            res[u] = 1
            return res
        assert(len(u)==A.shape[0])
        return np.array(u).reshape((-1))
    Xbase = vec_u(Xbase)
    Ybase = vec_u(Ybase)
    x0 = np.zeros(A.shape[0]) if x0 is None else x0
    retList = []
    for x in X:
        # get a second order equation a y^2 + by + c = 0
        a = Ybase.T @ A @ Ybase
        b = 2 * (x * Xbase.T + x0) @ A @ Ybase + B.T @ Ybase
        c = ((x * Xbase + x0).T ) @ A @ (Xbase * x + x0) + B.T @ (Xbase * x + x0) + C

        # print(a,b,c)

        if(abs(a) < 1e-6):
            retList.append((x,-c/b))
        else:
            delta = b**2 - 4 * a * c 
            if(delta<0):
                continue
            else:
                delta = np.sqrt(delta)
                retList.append((x,(-b + delta)/(2*a)))
                retList.append((x,(-b - delta)/(2*a)))
                # print((-b - delta)/(2*a))
    return np.array(retList)

def phasePortrait(traj,dim = 1,key = "Hx",ax = None,label = None):
    """
        given trajdict("Hx" is needed), return the phasePortrait plot
    """
    ax = plt.gca() if ax is None else ax
    label = dim if label is None else label
    lenth = len(traj[key][0])//2
    ax.plot(np.array(traj[key])[:,dim],np.array(traj[key])[:,dim+lenth],label = label)
    ax.legend()
    return ax

if __name__ == "__main__":
    mc = 10
    HA = np.zeros((14,14))
    HA[1,1] = mc
    HA[1,8] = HA[8,1] = 1

    # CBF of the body_x
    HA[0,0] = -mc * 50
    HA[0,7] = HA[7,0] = -1 * 50

    Hb = np.zeros(14)
    Hc = -mc * 0.5 * 0.5

    # pts = QuadContour(HA,Hb,Hc,np.arange(-0.1,0.1,0.005))
    pts = QuadContour(HA,Hb,Hc,[-0])
    # print(pts)
    plt.plot(pts[:,0], pts[:,1], ".")
    plt.ylim(ymin = 0)
    plt.show()