import sys
sys.path.append(".")
from bson import json_util
import json
from util.AcademicPlotStyle import *
from glob import glob
import os
import pickle as pkl

exps = [".exps/2020-05-28-12_24_15.json",".exps/2020-05-30-02_08_27.json"]

epochs = [it for e in exps for it in json.load(open(e,"r"))["IterationInfo"]]

avlengths = [e["averageTrajLength"] for e in epochs if "averageTrajLength" in e.keys()]

plt.plot(avlengths)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Mean Traj Length($s$)")
plt.title("Trend of the Mean Length of Traj Samples")
plt.savefig("doc/pics/learnWalk_meantime.png",bbox_inches = 'tight', pad_inches = 0.0)
plt.show()


# """
#     Test the mean period (May Have Some Problems)
# """
# from util.trajAnalysis import getMeanPeriod
# import matplotlib.pyplot as plt
# exps = ["data/learncbf/relabeling_2020-05-28-12_24_15","data/learncbf/relabeling_protect_2020-05-30-02_08_27"]
# trajs = [pkl.load(open(t,"rb")) for exp in exps for t in glob(os.path.join(exp,"ExceptionTraj*.pkl"))]
# Num_CBFs = [len(t["CBFCons"][-1]) for t in trajs]
# meanPeriods = [getMeanPeriod(t,tauInd=7) for t in trajs]
# plt.plot(Num_CBFs,meanPeriods,".",alpha = 0.5)
# plt.show()