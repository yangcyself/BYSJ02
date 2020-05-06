"""
Test the function `CBF_GEN_conic` in CBFCtrl.py
"""

import sys
sys.path.append(".")
import globalParameters as GP
GP.STARTSIMULATION = False
from ctrl.CBFCtrl import *

A,B,C = CBF_GEN_conic(7,10,(1,0.5,-3,1),(2,2,6,np.array([0,0,0,1,-1,0,0])))

print("A:\n",A)
print("B:\n",B)
print("C:\n",C)