"""
    Load a trajectory from data and simulate using mu or play it 
"""
import sys
sys.path.append(".")

from ctrl.playBackCtrl import *
import pickle as pkl

from util.visulization import QuadContour
from util.AcademicPlotStyle import *

import matplotlib.pyplot as plt
from util.visulization import phasePortrait
from glob import glob 
from learnCBF.FittingUtil import loadCBFsJson

# traj = pkl.load(open("data/Traj/1590638422.pkl","rb"))
traj = pkl.load(open(glob("data/learncbf/relabeling_2020-05-28-12_24_15/ExceptionTraj*.pkl")[-1],"rb"))
# traj["Hx"] = traj["Hx"][:-1000]
# traj = pkl.load(open("data/Traj/1588230703.pkl","rb"))
# traj = pkl.load(open("data/learncbf/SafeWalk2_2020-05-24-01_36_18/ExceptionTraj1590276882.pkl","rb"))
# traj = pkl.load(open("data/learncbf/SafeWalk2_2020-05-24-01_36_18/ExceptionTraj1590286056.pkl","rb"))
# traj = pkl.load(open(glob("data/learncbf/protected_2020-05-25-01_39_37/ExceptionTraj*.pkl")[-1],"rb"))
# traj = pkl.load(open("data/Traj/1590599280.pkl","rb"))# with
# traj = pkl.load(open("data/Traj/1590599448.pkl","rb"))# without
# CBFs = loadCBFsJson("data/learncbf/debug_2020-05-28-00_45_26/CBF1.json")

# traj = pkl.load(open("data/Traj/1588230736.pkl","rb"))
# traj = pkl.load(open("data/Traj/1588230836.pkl","rb"))

# for i,name in enumerate(["x","y",'r',"q1_r","q2_r","q1_l","q2_l","stance_toe","swing_toe","tau"]):
# for i,(name,dname) in enumerate(zip(["$p_x$","$p_y$",'$r$',"$q_1$","$q_2$","$q_3$","$q_4$","$r_{stance}$","$r_{swing}$","$\\tau$"],
#                                     ["$\dot{p}_x$","$\dot{p}_y$",'$\dot{r}$',"$\dot{q}_1$","$\dot{q}_2$","$\dot{q}_3$","$\dot{q}_4$","$\dot{r}_{stance}$","$\dot{r}_{swing}$","$\dot{\\tau}$"])):
for i,(name,dname) in enumerate(zip(["$p_x$","$p_y$",'$r$',"$\hat{q}_1$","$\hat{q}_2$","$\hat{q}_3$","$\hat{q}_4$","$\\tau$"],
                                    ["$\dot{p}_x$","$\dot{p}_y$",'$\dot{r}$',"$\dot{\hat{q}}_1$","$\dot{\hat{q}}_2$","$\dot{\hat{q}}_3$","$\dot{\hat{q}}_4$","$\dot{\\tau}$"])):
    if(i ==0):
        continue
    print(i)
    plt.figure()
    phasePortrait(traj,dim = i,label=name)
    plt.xlabel(name+("(rad)" if i <2 else "(m)"))
    plt.ylabel(dname+("(rad/s)" if i <2 else "(m/s)"))
    plt.grid()
    plt.savefig("doc/pics/phase-2020531/learnWalk%s.png"%(name[-4:-1]),bbox_inches = 'tight', pad_inches = 0.1)
    plt.draw()
    # break

## construst simulation with and without CBF
# plt.figure()
# x0 = np.mean(traj["Hx"], axis = 0)
# pts = QuadContour(*(CBFs[1]), np.arange(-1,1,0.01),4,14, x0 = x0)
# plt.plot(pts[:,0], pts[:,1],label = "$B_{1\_safe}$",c = "b")
# plt.legend()
# plt.plot([0,0],[-50,30], c = sns.xkcd_rgb["greyish"], label = "$B_{0\_safe}$")
# phasePortrait(traj,dim = 4,label="Traj Without CBF",c="g")
# plt.ylim((-20,30))
# plt.xlim((-1,2))
# plt.xlabel("$q_2$")
# plt.ylabel("$\dot{q}_2$")
# plt.grid()
# plt.title("Simulation result without CBF")
# plt.savefig("doc/pics/learnq2_withoutCBF.png",bbox_inches = 'tight', pad_inches = 0.1)
# plt.draw()

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