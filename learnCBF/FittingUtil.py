import numpy as np

class kernel:
    """
        a map to some higher dim space. E.g (x,y) -> (x^2,y^2,x,y)
    """
    @staticmethod
    def augment(x):
        """
            input is either 1dim dataponit, or a matrix where each row is a datapoint
            return the augmented matrix, where each row is the feature of a datapoint
        """
        x = np.array(x)
        if(x.ndim==1):
            x  = x.reshape((1,-1))
        x = x.T
        return np.concatenate([[a*b for i,a in enumerate(x) for b in x[i:] ],x,np.ones((1,x.shape[1]))],axis = 0).T

    @staticmethod
    def jac(x):
        """
            input is 1dim dataponit
            return the jacobian of this datapoint
        """
        x = np.array(x)
        return np.concatenate([ [ [(a if k==j+i else 0) + (b if k==i else 0)  for k in range(len(x))] 
                                for i,a in enumerate(x) for j,b in enumerate(x[i:]) ] ,np.eye(len(x)),np.zeros((1,len(x)))],axis=0)
    
    @staticmethod
    def GetParam(w,dim=20):
        """
            input the trained w
            Return the A, b, c of x^TAx + b^T x + c
        """
        # print("w:",w)
        A = np.array([[w[min(i,j)*dim - int(min(i,j)*(min(i,j)-1)/2) - min(i,j) + max(i,j)]/(1 if(i==j)else 2)  for j in range(dim)] for i in range(dim)])
        b = w[-dim-1:-1]
        c = w[-1]
        return A,b,c
