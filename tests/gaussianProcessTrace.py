"""
The evolution of the p, mu, std etc of the gaussian process
"""
import matplotlib.pyplot as plt
import dill as pkl
import numpy as np

# GPProb = pkl.load(open("data\gaussian\GPProbTrace2020-05-10-14_48_27.pkl","rb"))
GPProb = pkl.load(open("data\gaussian\GPProbTrace2020-05-11-02_30_11.pkl","rb"))

safeTrace = []
DangerTrace = []
ind_count = 0
TrajStarts = []
for safeTraj,DangerTraj in GPProb:
    TrajStarts.append(ind_count)
    safeTrace += [(ind_count+i,*t) for i, t in enumerate(safeTraj)]
    ind_count += len(safeTraj)
    DangerTrace += [(ind_count+i,*t) for i, t in enumerate(DangerTraj)]
    ind_count += len(DangerTraj)
    

GPProb = np.array(GPProb)
safeTrace,DangerTrace = np.array(safeTrace),np.array(DangerTrace)
def addTrajStarts(y =(lambda x: 0)):
    ax = plt.gca()
    ax.plot(TrajStarts, [y(x) for x in TrajStarts],".",label = "Traj Starts")


def plotprob(trace):
    plt.figure()
    plt.plot(trace[:,0])
    plt.draw()
    plt.figure()
    plt.plot(trace[:,0], trace[:,1],".",markersize = 1,label = "prob")
    addTrajStarts()
    plt.title("prob")
    plt.legend()
    plt.grid()
    plt.draw()

def plotmu(trace):
    plt.figure()
    plt.plot(trace[:,0], trace[:,2],".",markersize = 1,label = "mu")
    addTrajStarts()
    plt.title("mu")
    plt.legend()
    plt.grid()
    plt.draw()


def plotstd(trace):
    plt.figure()
    plt.plot(trace[:,0], trace[:,3],".",markersize = 1, label = "std")
    addTrajStarts()
    plt.title("std")
    plt.legend()
    plt.grid()
    plt.draw()

# plotprob(safeTrace)
# plotstd(safeTrace)
# plotmu(safeTrace)

plotprob(DangerTrace)
plotstd(DangerTrace)
plotmu(DangerTrace)


plt.show()