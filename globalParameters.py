import numpy as np


## Experiment settings

STARTSIMULATION = True # whether start the pybullet simulation

GUIVIS = True # whether open GUI

DT = 1e-3

# URDF_FILE = "five_link_walker_yz.urdf"
# PRISMA_AXIS = np.array([1,2])
# CROSS_SIGN = 1

URDF_FILE = "five_link_walker_xz.urdf"
PRISMA_AXIS = np.array([0,2])
CROSS_SIGN = -1

MU = 0.6

## common definitions

QDIMS = 7

robot = None

floor = None
numJoints = None

MAXTORQUE = np.array([0,0,0,500,500,500,500])

GRAVITY = -9.8

CATCH_KEYBOARD_INTERRPUT = True

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
    def x(self):
        return 0
    @property
    def y(self):
        return 1
    @property
    def r(self):
        return 2
    @property
    def q1_r(self):
        return 3
    @property
    def q2_r(self):
        return 4
    @property
    def q1_l(self):
        return 6
    @property
    def q2_l(self):
        return 7
            
qind = qind_t()

KEYBOARD_INTERRPUT = False