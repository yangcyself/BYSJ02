"""
Plot the mu and std of a series of samples according to the gaussian process
"""

import sys
sys.path.append(".")
import globalParameters as GP
GP.STARTSIMULATION = False

from ctrl.playBackCtrl import *
import dill as pkl
import matplotlib.pyplot as plt


# traj = pkl.load(open("data/Traj/1588045936.pkl","rb"))
traj = pkl.load(open("data/Traj/1589081488.pkl","rb"))
# traj = pkl.load(open("data/Traj/1588230736.pkl","rb"))
# traj = pkl.load(open("data/Traj/1588230836.pkl","rb"))

G = pkl.load(open("data/gaussian/2020-05-10-09_03_08.pkl","rb"))



mu_std_list = [G(t) for t in traj["Hx"]]
t_list = traj["t"]
mu_list = np.array([m[0] for m in mu_std_list])
std_list = np.array([m[1][0] for m in mu_std_list])

mu_upper_list = mu_list + std_list
mu_lower_list = mu_list - std_list
plt.plot(t_list, mu_list)
plt.plot(t_list, mu_upper_list,c="g",alpha = 0.7)
plt.plot(t_list, mu_lower_list,c="g",alpha = 0.7)
plt.savefig(os.path.join("data/pics/guassianProcessTest", "%s.png"%str(int(time.time()))))
plt.show()