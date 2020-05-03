import sys
sys.path.append(".")
import globalParameters as GP
GP.GUIVIS = False
GP.CATCH_KEYBOARD_INTERRPUT = False
from ctrl.CBFWalker import *
import matplotlib.pyplot as plt
from ExperimentSecretary.Core import Session
from util.visulization import QuadContour
import pickle as pkl


Balpha =  np.array([
    [-0.160827987125513,	-0.187607134811070,	-0.0674336818331384,	-0.125516015984117,	-0.0168214255686846,	-0.0313250368138578],
    [0.0873653371709151,	0.200725783096372,	-0.0300252126874686,	0.148275517821605,	0.0570910838662556,	0.209150474872096],
    [-0.0339476931185450,	-0.240887593110385,	-0.840294819928089,	-0.462855953941200,	-0.317145689802212,	-0.157521190052112],
    [0.207777212951830,	    0.519994705831362,	2.12144766616269,	0.470232658089196,	0.272391319931093,	0.0794930524259044] ])

Balpha[[0,2],:] += np.math.pi

def reset(ct):
    ct.resetStatic()
    ct.setState([
    -0.0796848040543580,
    0.795256332708201,
    0.0172782890029275,
    -0.160778382265038 + np.math.pi,
    0.0872665521987388,
    -0.0336100968515576 + np.math.pi,
    0.207826470055357,

    0.332565019796099,
    0.0122814228026706,
    0.100448183395960,
    -0.284872086257488,
    1.19997002232031,
    -2.18405395105742,
    3.29500361015889,
    ])

def SampleTraj(BalphaStd,Balpha=Balpha):
    ct = CBF_WALKER_CTRL()
    reset(ct)
    Balpha = Balpha + BalphaStd*(0.5-np.random.random(size = Balpha.shape))
    Kp = 200 + 800*np.random.rand()
    Kd = np.log(Kp)*np.random.rand()*4
    CBF_WALKER_CTRL.IOcmd.reg(ct,Balpha = Balpha)
    CBF_WALKER_CTRL.IOLin.reg(ct,kp = Kp, kd = Kd)
    Traj = []
    def callback_traj(obj):
        Traj.append((obj.t, obj.state, obj.Hx, obj.CBF_CLF_QP))
    ct.callbacks.append(callback_traj)
    def callback_break(obj):
        return not (3*np.math.pi/4 < obj.Hx[7] < 5*np.math.pi/4 
                    and 3*np.math.pi/4 < obj.Hx[8] < 5*np.math.pi/4
                    and -0.1<obj.Hx[0] and obj.LOG_CBF_SUCCESS)
    ct.callbacks.append(callback_break)
    ct.step(5)
    return Traj


def storTraj(Traj,path):
    dumpname = os.path.abspath(os.path.join(path,"%d.pkl"%time.time()))
    pkl.dump({
        "t": [t[0] for t in Traj],
        "state": [t[1] for t in Traj],
        "u": [t[3] for t in Traj]
    } ,open(dumpname,"wb"))

def storSample(SafeSamples,DangerSamples,path):
    dumpname = os.path.abspath(os.path.join(path,"%d.pkl"%time.time()))
    pkl.dump({"safe":SafeSamples,"danger":DangerSamples}, open(dumpname,"wb"))

class SampleSession_t(Session):
    """
        define the Session class to record the information
    """
    def __init__(self, BalphaStd = 0.03, Ntraj = 100, Nsample = 10, T_Threshold = 1.3):
        super().__init__(expName="IOWalkSample")
        self.BalphaStd = BalphaStd
        self.T_Threshold = T_Threshold
        self.storTrajPath = "./data/Traj/IOWalkSample/%s"%(self._storFileName)
        self.storSamplePath = "./data/StateSamples/IOWalkSample/%s"%(self._storFileName)
        os.makedirs(self.storTrajPath,exist_ok=True)
        os.makedirs(self.storSamplePath,exist_ok=True)
        self.Nsamples = Nsample
        self.Ntraj = Ntraj
        self.TrajLengths = []
        self.add_info("BalphaStd",self.BalphaStd)
        self.add_info("storTrajPath",self.storTrajPath)
        self.add_info("storSamplePath",self.storSamplePath)
        self.add_info("Nsamples",self.Nsamples)
        self.add_info("Ntraj",self.Ntraj)
        self.add_info("T_Threshold", self.T_Threshold)


    def body(self):
        self.trajCount = 0
        self.sampleCount_safe = 0
        self.sampleCount_danger = 0
        for i in range(self.Ntraj):
            Traj = SampleTraj(self.BalphaStd)

            T_list = np.array([t[0] for t in Traj])
            Hx_list = np.array([t[2] for t in Traj])

            T_whole = T_list[-1]
            if(T_whole < self.T_Threshold):
                continue

            self.TrajLengths.append(T_whole) 
            storTraj(Traj, self.storTrajPath)
            self.trajCount += 1

            # sample states from the traj
            # [0, T_safe] are times for safe samples, [T_danger, T_whole] are times for danger samples
            T_safe = T_whole - 1
            T_danger = T_whole - 0.5
            ind_safe = np.argmin(abs(T_list-T_safe))
            ind_danger = np.argmin(abs(T_list-T_danger))
            
            if(ind_safe < self.Nsamples):
                SafeSamples = Hx_list[:ind_safe]
            else:
                SafeSamples = Hx_list[range(0,ind_safe,ind_safe//self.Nsamples)]
            self.sampleCount_safe += len(SafeSamples)
            if(len(Traj) - ind_danger < self.Nsamples):
                DangerSamples = Hx_list[ind_danger:]
            else:
                DangerSamples = Hx_list[range(ind_danger,len(Traj),(len(Traj) - ind_danger)//self.Nsamples)]
            storSample(SafeSamples,DangerSamples,self.storSamplePath)
            self.sampleCount_danger += len(DangerSamples)


class SampleSession(SampleSession_t):
    @SampleSession_t.column
    def MeanTrajLength(self):
        return np.mean(self.TrajLengths)
    @SampleSession_t.column
    def StdTrajLength(self):
        return np.std(self.TrajLengths)
    @SampleSession_t.column
    def trajCount(self):
        return self.trajCount
    @SampleSession_t.column
    def sampleCount_safe(self):
        return self.sampleCount_safe
    @SampleSession_t.column
    def sampleCount_danger(self):
        return self.sampleCount_danger

CBF_WALKER_CTRL.restart()
            
if __name__ == '__main__':
    s = SampleSession(Ntraj=1000)
    s()
    