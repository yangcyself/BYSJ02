import numpy as np
import matplotlib.pyplot as plt

def QuadContour(A,B,C, X, Xbase = np.array([1,0,0,0,0,0,0, 0,0,0,0,0,0,0,]), Ybase = np.array([0,1,0,0,0,0,0, 0,0,0,0,0,0,0,])):
    """
    return a list of points (x,y) where X is from the set list X

    args: Xbase, Ybase are the axis to be plotted, We assume the value of all other spaces are 0
    """
    retList = []
    for x in X:
        # get a second order equation a y^2 + by + c = 0
        a = Ybase.T @ A @ Ybase
        b = 2 * x * Xbase.T @ A @ Ybase + B.T @ Ybase
        c = x * (Xbase.T @ A @ Xbase * x + B.T @ Xbase) + C

        print(a,b,c)

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
                print((-b - delta)/(2*a))
    return np.array(retList)

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