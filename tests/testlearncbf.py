import sys
sys.path.append(".")
from learnCBF.IOWalk.learncbf import *
from util.CBF_builder import CBF_GEN_degree0


if __name__ == '__main__':
        # import learnCBF.IOWalk.learncbf
    # learnCBF.IOWalk.learncbf.SYMMETRY_AUGMENT = False
    # s = LearnCBFSession([CBF_GEN_degree0(10,(0,1,0,4)) # leg limit 
    #                      ] , lin_eps = 10,
    #     name = "debug",numSample=20, Iteras = 1, dangerDT=0.01, safeDT=0.1,
    #     class_weight={1:1, -1:1})
    # s()
    # exit()
    
    # A,b,c,_,samples = learnCBFIter(CBF_GEN_conic(10,100000,(0,1,0.1,4)),[], 
    #                 dim=20, mc = 0.01, gamma0=0.01, gamma = 1, gamma2=1, class_weight = None,
    #                 numSample = 20, dangerDT=0.01, safeDT = 0.5)

    # dumpJson(A,b,c,"data/CBF/LegAngletmp.json")
    
    from util.visulization import QuadContour
    from util.AcademicPlotStyle import *
    import matplotlib.pyplot as plt
    import json
    from learnCBF.FittingUtil import loadCBFsJson

    # CBFs =loadCBFsJson("./data/learncbf/redLegQ1_2020-05-19-09_12_04/CBF1.json") # trained with 500 samples
    # CBFs = loadCBFsJson("data/learncbf/debug_2020-05-27-19_47_13/CBF1.json")
    CBFs = loadCBFsJson("data/learncbf/debug_2020-05-28-00_45_26/CBF1.json")
    # CBFs = loadCBFsJson("data/learncbf/debug_2020-05-27-21_37_49/CBF1.json")

    # CBFs = loadCBFsJson("data/learncbf/SafeWalk2_2020-05-23-00_10_03/CBF1.json")
    # Polyparameter = json.load(open("data/CBF/LegAngletmp.json","r")) # trained with 500 samples
    


    # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    # samples = pkl.load(open("data/learncbf/debug_2020-05-23-10_49_06/samples0.pkl","rb"))
    # samples = pkl.load(open("data/learncbf/debug_2020-05-27-19_47_13/samples0.pkl","rb"))
    samples = pkl.load(open("data/learncbf/debug_2020-05-28-00_45_26/samples0.pkl","rb"))
    # samples = pkl.load(open("data/learncbf/debug_2020-05-27-21_37_49/samples0.pkl","rb"))

    Xp = np.array([x for danger_s, safe_s in samples for x,u,DB in safe_s])
    Xn = np.array([x for danger_s, safe_s in samples for x,u,DB in danger_s])

    # x0 = np.mean(np.array(list(Xp)+list(Xn)), axis = 0)
    x0 = np.mean(np.array(list(Xp)), axis = 0)
    x0[[4,14]] = 0
    x0 = x0.reshape(-1)

    plt.figure(figsize=(8,6))
    for (HA_CBF,Hb_CBF,Hc_CBF) in CBFs[1:]:
        print("x0 :",x0)
        print("x0.T@HA_CBF@x0 :",x0.T@HA_CBF@x0)
        pts = QuadContour(HA_CBF,Hb_CBF,Hc_CBF, np.arange(-1,1,0.01),4,14, x0 = x0)
        plt.plot(pts[:,0], pts[:,1],label = "$B_{1\_safe}$",c = "b")
        plt.legend()
    plt.plot([0,0],[-50,20], c = sns.xkcd_rgb["greyish"], label = "$B_{0\_safe}$")

    # plt.draw()
    # plt.figure()

    Xn0 = np.array([x for danger_s, safe_s in samples for x,u,DB in danger_s[:1]])
    plt.plot(Xp[:,4],Xp[:,14],".",c = "g",label = "$X_{safe}$")
    plt.plot(Xn[:,4],Xn[:,14],".",c = "r",label = "$X_{danger}$")
    plt.plot(Xn0[:,4],Xn0[:,14],".",c = "k",label = "$X_{violated}$")
    plt.ylim((-50,20))
    plt.xlim((-0.1,1))
    plt.xlabel("$q_2$")
    plt.ylabel("$\dot{q}_2$")
    legendParam["fontsize"] = 18
    plt.legend(**legendParam)
    plt.grid()
    plt.title("CBF sampling and fitting of $q_2$")
    plt.draw()
    plt.savefig("doc/pics/learnq2.png",bbox_inches = 'tight', pad_inches = 0.3)
    plt.show()