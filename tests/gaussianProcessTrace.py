"""
The evolution of the p, mu, std etc of the gaussian process
"""
import matplotlib.pyplot as plt
import dill as pkl
import numpy as np

GPProb = pkl.load(open("data\gaussian\GPProbTrace2020-05-10-14_48_27.pkl","rb"))

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

def addTrajStarts(y =(lambda x: 0)):
    ax = plt.gca()
    ax.plot(TrajStarts, [y(x) for x in TrajStarts],".",label = "Traj Starts")


plt.figure()
plt.plot(GPProb[:,0], GPProb[:,1],label = "prob")
addTrajStarts()
plt.title("prob")
plt.legend()
plt.grid()
plt.draw()

plt.figure()
plt.plot(GPProb[:,0], GPProb[:,2],label = "mu")
addTrajStarts()
plt.title("mu")
plt.legend()
plt.grid()
plt.draw()

plt.figure()
plt.plot(GPProb[:,0], GPProb[:,3],label = "std")
addTrajStarts()
plt.title("std")
plt.legend()
plt.grid()
plt.draw()

plt.show()