import numpy as np


## Experiment settings

GUIVIS = True

## common definitions

QDIMS = 7

robot = None

floor = None
numJoints = None

MAXTORQUE = np.array([0,0,0,500,500,500,500])

class qind_t:
    def __init__(self):
        # convert from index of [x,y,r,q1,q2,q3,q4] to the indexes in the robot tree(includes toe)
        self.ind = [0,1,2,3,4,6,7]
    def __call__(self,i):
        return self.ind[i]
    def __iter__(self):
        for i in self.ind:
            yield i
    def __getitem__(self,i):
        return self.ind[i]

    @property
    def x():
        return 0
    @property
    def y():
        return 1
    @property
    def r():
        return 2
    @property
    def q1_r():
        return 3
    @property
    def q2_r():
        return 4
    @property
    def q1_l():
        return 6
    @property
    def q2_l():
        return 7
            
qind = qind_t()