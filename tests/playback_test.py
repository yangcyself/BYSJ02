"""
    Load a trajectory from data and simulate using mu or play it 
"""
import sys
sys.path.append(".")

from ctrl.playBackCtrl import *
import pickle as pkl
import matplotlib.pyplot as plt
from util.visulization import phasePortrait
from glob import glob 
# traj = pkl.load(open("data/Traj/1588045936.pkl","rb"))
# traj = pkl.load(open("data/Traj/1588230703.pkl","rb"))
# traj = pkl.load(open("data/learncbf/SafeWalk2_2020-05-24-01_36_18/ExceptionTraj1590276882.pkl","rb"))
# traj = pkl.load(open("data/learncbf/SafeWalk2_2020-05-24-01_36_18/ExceptionTraj1590286056.pkl","rb"))
traj = pkl.load(open(glob("data/learncbf/protected_2020-05-25-01_39_37/ExceptionTraj*.pkl")[-1],"rb"))
# traj = pkl.load(open("data/Traj/1588230736.pkl","rb"))
# traj = pkl.load(open("data/Traj/1588230836.pkl","rb"))

for i,name in enumerate(["x","y",'r',"q1_r","q2_r","q1_l","q2_l","stance_toe","swing_toe","tau"]):
    print("i")
    plt.figure()
    phasePortrait(traj,dim = i,label=name)
    plt.draw()

# ct =playBackCTRL(traj)
# ct.restart()
# ct.setStateT()

# Traj = []
# def callback_traj(obj):
#     Traj.append((obj.t, obj.state))
# ct.callbacks.append(callback_traj)

# # ct.step(ct.trajLength)

# time.sleep(1)

# ct.playstate(ct.trajLength)

# T = [t[0] for t in Traj]
# playbackState = np.array([ct.getTraj(t)['state'] for t in T])
# simuState = np.array([t[1] for t in Traj])

# dim = 5
# plt.figure()
# plt.plot(T,playbackState[:,dim],label = "playbackState")
# plt.plot(T,simuState[:,dim],label = "simuState")
# plt.legend()
plt.show()