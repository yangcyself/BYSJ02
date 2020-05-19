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

    # CTRL.restart()
    # A,b,c,_,samples = learnCBFIter(CBF_GEN_conic(10,100000,(0,1,0.1,4)),[], 
    #                 dim=20, mc = 0.01, gamma0=0.01, gamma = 1, gamma2=1, class_weight = None,
    #                 numSample = 20, dangerDT=0.01, safeDT = 0.5)

    # dumpJson(A,b,c,"data/CBF/LegAngletmp.json")
    
    from util.visulization import QuadContour
    import matplotlib.pyplot as plt
    import json

    # pts = QuadContour(*CBF_GEN_conic(10,100000,(0,1,0.1,4)), np.arange(-0,0.015,0.0000001),4,14)
    # plt.plot(pts[:,0], pts[:,1], ".", label = "CBF")
    # # plt.ylim(ymin = 0)
    # plt.title("x,y trajectory with CBF")

    # plt.draw()
    # plt.figure()

    # Polyparameter = json.load(open("./data/learncbf/redLegQ2_2020-05-14-15_16_18/CBF1.json","r")) # trained with 500 samples
    Polyparameter = json.load(open("data/CBF/LegAngletmp.json","r")) # trained with 500 samples
    HA_CBF,Hb_CBF,Hc_CBF = np.array(Polyparameter["A"]), np.array(Polyparameter["b"]), Polyparameter["c"]

    print("HA_CBF[4][4]",HA_CBF[4][4])
    print("HA_CBF[14][14] :",HA_CBF[14][14])
    print("HA_CBF[4][14] :",HA_CBF[4][14])
    print("Hb_CBF[4] :",Hb_CBF[4])
    print("Hb_CBF[14] :",Hb_CBF[14])
    print("Hc_CBF :",Hc_CBF)
    


    # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    samples = pkl.load(open("./data/learncbf/redLegQ2_2020-05-14-15_16_18/samples0.json","rb"))
    Xp = np.array([x for danger_s, safe_s in samples for x,u,DB in safe_s])
    Xn = np.array([x for danger_s, safe_s in samples for x,u,DB in danger_s])

    # x0 = np.mean(np.array(list(Xp)+list(Xn)), axis = 0)
    x0 = np.mean(np.array(list(Xp)), axis = 0)
    x0[[4,14]] = 0
    x0 = x0.reshape(-1)
    print("x0 :",x0)
    print("x0.T@HA_CBF@x0 :",x0.T@HA_CBF@x0)
    
    pts = QuadContour(HA_CBF,Hb_CBF,Hc_CBF, np.arange(-50,50,0.01),4,14, x0 = x0)
    plt.plot(pts[:,0], pts[:,1], ".", label = "New CBF")
    plt.legend()

    # plt.draw()
    # plt.figure()

    plt.plot(Xp[:,4],Xp[:,14],".",c = "g",label = "safe points")
    plt.plot(Xn[:,4],Xn[:,14],".",c = "r",label = "danger points")
    plt.legend()
    # plt.ylim((-50,50))
    plt.draw()

    plt.figure()
    # samples = pkl.load(open("./data/learncbf/tmp.pkl","rb"))
    samples = pkl.load(open("./data/learncbf/redLegQ2_2020-05-14-15_16_18/samples0.json","rb"))
    X = [x for danger_s, safe_s in samples for x,u,DB in danger_s+safe_s]
    y_list = [i for danger_s, safe_s in samples for i in ([-1]*len(danger_s)+[1]*len(safe_s))]
    # print("X :",X)
    # print("y_list :",y_list)
    
    
    Xp = np.array([x for x,y in zip(X,y_list) if y==1])
    Xn = np.array([x for x,y in zip(X,y_list) if y==-1])
    plt.plot(Xp[:,4],Xp[:,14],".",c = "g",label = "safe points")
    plt.plot(Xn[:,4],Xn[:,14],".",c = "r",label = "danger points")
    plt.legend()
    # plt.ylim((-50,50))
    plt.draw()

    
    plt.show()