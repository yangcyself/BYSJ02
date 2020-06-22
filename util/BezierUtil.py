import numpy as np
from math import factorial


def nchoosek(n,k):
    return factorial(n) / (factorial(k)*factorial(n-k))


def evalBezier(Balpha,tau):
    """
        This Bezier calculation follows the routine in [TA_GaitDesign](https://github.com/yangcyself/TA_GaitDesign)
    """
    K,M = Balpha.shape # K: The number of the virtual constraint, M: the order
    M = M-1
    x = np.sum(
        [tau**m * (1-tau)**(M-m) * 
            Balpha[:,m] * nchoosek(M,m) for m in range(M+1)], 
        axis = 0)
    dx = np.sum(
        [M * nchoosek(M-1,m) * tau**(m) * (1-tau)**(M-1-m) * 
            (Balpha[:,m+1]-Balpha[:,m]) for m in range(M)], 
        axis = 0)

    ddx = np.sum(
        [M * (M-1) * nchoosek(M-2,m) * tau**(m) * (1-tau)**(M-2-m) * 
            (Balpha[:,m] - 2*Balpha[:,m+1] + Balpha[:,m+2])  for m in range(M-1)], 
        axis = 0)
    return x,dx,ddx


def loadCSV(file_name):
    return np.genfromtxt(file_name, delimiter=',')