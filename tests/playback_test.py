"""
    Load a trajectory from data and simulate using mu or play it 
"""
import sys
sys.path.append(".")

from ctrl.playBackCtrl import *
import pickle as pkl
import matplotlib.pyplot as plt

# traj = pkl.load(open("data/Traj/1588045936.pkl","rb"))
traj = pkl.load(open("data/Traj/1588230703.pkl","rb"))
# traj = pkl.load(open("data/Traj/1588230736.pkl","rb"))
# traj = pkl.load(open("data/Traj/1588230836.pkl","rb"))

ct =playBackCTRL(traj)
ct.restart()
ct.setStateT()

Traj = []
def callback_traj(obj):
    Traj.append((obj.t, obj.state))
ct.callbacks.append(callback_traj)

ct.step(1)

time.sleep(1)

ct.playstate(1)

T = [t[0] for t in Traj]
playbackState = np.array([ct.getTraj(t)['state'] for t in T])
simuState = np.array([t[1] for t in Traj])

dim = 5
plt.plot(T,playbackState[:,dim],label = "playbackState")
plt.plot(T,simuState[:,dim],label = "simuState")
plt.legend()
plt.show()