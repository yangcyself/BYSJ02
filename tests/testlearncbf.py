import sys
sys.path.append(".")
from learnCBF.IOWalk.learncbf import *


if __name__ == '__main__':
    
    ### Hand craft test data
    # Xp_ = np.array([[2,-1],[3,-4],[2,-3],[1,-2]])
    # Xn_ = np.array([[1,-10],[2,-10],[3,-10],[4,-10]])
    # Xp = np.zeros((4,20))
    # Xp[:,[4,14]] = Xp_
    # Xn = np.zeros((4,20))
    # Xn[:,[4,14]] = Xn_
    # samples = [([(x,np.zeros(7),(None,None,None)) for x in Xn],[(x,np.zeros(7),(None,None,None)) for x in Xp])]
    # pkl.dump(samples, open("./data/learncbf/Handtmp.pkl","wb"))

    # import learnCBF.IOWalk.learncbf
    # learnCBF.IOWalk.learncbf.SYMMETRY_AUGMENT = False
    # s = LearnCBFSession([CBF_GEN_conic(10,10000,(0,1,0.1,4)), # leg limit 
    #                      ] ,
    #     name = "debug",numSample=20, Iteras = 2, dangerDT=0.01, safeDT=0.1,
    #     class_weight={1:1, -1:1})
    # s()
    # exit()
    
    # A,b,c,_,samples = learnCBFIter(CBF_GEN_conic(10,100000,(0,1,0.1,4)),[], 
    #                 dim=20, mc = 0.01, gamma0=0.01, gamma = 1, gamma2=1, class_weight = None,
    #                 numSample = 20, dangerDT=0.01, safeDT = 0.5)

    # dumpJson(A,b,c,"data/CBF/LegAngletmp.json")
    
    from util.visulization import QuadContour
    import matplotlib.pyplot as plt
    import json
    from learnCBF.FittingUtil import loadCBFsJson

    # pts = QuadContour(*CBF_GEN_conic(10,100000,(0,1,0.1,4)), np.arange(-0,0.015,0.0000001),4,14)
    # plt.plot(pts[:,0], pts[:,1], ".", label = "CBF")
    # # plt.ylim(ymin = 0)
    # plt.title("x,y trajectory with CBF")

    # plt.draw()
    # plt.figure()

    # CBFs =loadCBFsJson("./data/learncbf/redLegQ1_2020-05-19-09_12_04/CBF1.json") # trained with 500 samples
    CBFs = loadCBFsJson("data/learncbf/debug_2020-05-23-10_49_06/CBF2.json")
    # CBFs = loadCBFsJson("data/learncbf/SafeWalk2_2020-05-23-00_10_03/CBF1.json")
    # Polyparameter = json.load(open("data/CBF/LegAngletmp.json","r")) # trained with 500 samples
    


    # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    # samples = pkl.load(open("data/learncbf/debug_2020-05-23-10_49_06/samples0.pkl","rb"))
    samples = pkl.load(open("data/learncbf/debug_2020-05-23-10_49_06/samples0.pkl","rb"))
    Xp = np.array([x for danger_s, safe_s in samples for x,u,DB in safe_s])
    Xn = np.array([x for danger_s, safe_s in samples for x,u,DB in danger_s])

    # x0 = np.mean(np.array(list(Xp)+list(Xn)), axis = 0)
    x0 = np.mean(np.array(list(Xp)), axis = 0)
    x0[[4,14]] = 0
    x0 = x0.reshape(-1)

    for (HA_CBF,Hb_CBF,Hc_CBF) in CBFs:
        print("x0 :",x0)
        print("x0.T@HA_CBF@x0 :",x0.T@HA_CBF@x0)
        pts = QuadContour(HA_CBF,Hb_CBF,Hc_CBF, np.arange(-1,1,0.01),4,14, x0 = x0)
        plt.plot(pts[:,0], pts[:,1], ".", label = "New CBF")
        plt.legend()

    # plt.draw()
    # plt.figure()

    plt.plot(Xp[:,4],Xp[:,14],".",c = "g",label = "safe points")
    plt.plot(Xn[:,4],Xn[:,14],".",c = "r",label = "danger points")
    plt.legend()
    plt.ylim((-100,100))
    plt.draw()

    plt.show()