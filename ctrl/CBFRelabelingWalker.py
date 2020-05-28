"""
The combination of CBF and IOlinearization controller. 
    The linearization forms the "CLF", and some simple CBF condition is set in the CBF-QP
"""

import sys
sys.path.append(".")

from ctrl.CBFWalker import *


class CBF_Relabeling_WALKER_CTRL(CBF_WALKER_CTRL):
    """
        The state Hx is maintained with relabeling. specifically: q_1, q_2 always for stance leg, q_3, q_4 always for swing leg.
    """

    Hxdim = 8
    
    @CTRL_COMPONENT
    def CBF_Ftr_Mat(self):
        """
        (Map, reverse_Map)
            NOTE: For current implementation, the Feature also uses state[0] as the theta of swing leg
                See `VC_c` in `IOLinearCtrl` for detail
            
            One state is added to the seven: Tau
        """
        redStance, theta_plus, Labelmap = self.STEP_label
        leg_angle_mat = np.array([1,0.5]) # the q_1 + 0.5 q_2
        Mat = np.concatenate([
                np.concatenate([np.eye(7),np.zeros((2+7,7))],axis = 0),
                np.concatenate([np.zeros((7+1,7)), np.eye(7), np.zeros((1,7))],axis = 0)
        ],axis = 1)
        Mat[3:7,3:7] = Mat[11:15,10:14] = Labelmap
        rMat = Mat.copy().T
        return Mat, rMat

