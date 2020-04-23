import sys
sys.path.append(".")
from ctrl.WBCCtrl import *
CTRL.restart()
# ct = WBC_CTRL()
ct = QP_WBC_CTRL()
ct.step(10)