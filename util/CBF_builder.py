import numpy as np

########################################
### UTIL FUNCTION FOR GENERATING CBF ###
########################################

def CBF_GEN_conic(dim,mc,*args:(float,float,float,{float, np.array})):
    """
    Add  u a u.T + b.T u + c to the CBF
    arg dim: int     The total dimension of the state, CBF matrix is 2dim x 2dim
    arg mc:          The 'gamma' of the CBF matrix
    arg *args: [(a,b,c,u)] x n 
    arg a,b,c:       The factor 
    arg u:         union{int, np.array([dim])} The dimension of the constraint
    """

    def vec_u(u):
        if(type(u)==int):
            res = np.zeros((dim,1))
            res[u] = 1
            return res
        assert(len(u)==dim)
        return np.array(u).reshape((-1,1))
    args = [(a,b,c,vec_u(u)) for (a,b,c,u) in args]

    HA_CBF = np.zeros((2*dim,2*dim))
    HA_CBF[:dim,:dim] = np.sum(
        a * mc * u @ u.T for (a,b,c,u) in args)
    HA_CBF[:dim,dim:] = HA_CBF[dim:,:dim] = np.sum(
        a * u @ u.T for (a,b,c,u) in args)

    Hb_CBF = np.zeros(2*dim)
    Hb_CBF[:dim] = np.sum( mc * b * u.reshape(-1) for (a,b,c,u) in args)
    Hb_CBF[dim:] = np.sum( b * u.reshape(-1) for (a,b,c,u) in args)

    Hc_CBF = np.sum(mc * c for (a,b,c,u) in args)
    
    return HA_CBF,Hb_CBF,Hc_CBF


def CBF_GEN_degree1(dim, *args:(float,float,float,{int, np.array})):
    """
    Add constraint about velocity: u [0,a;0,a] u.T + [0;b].T u + c to the CBF
    arg dim: int     The total dimension of the state, CBF matrix is 2dim x 2dim
    arg *args: [(a,b,c,u)] x n 
    arg a,b,c:       The factor 
    arg u:         union{int, np.array([dim])} The dimension of the constraint
    """
    def vec_u(u):
        if(type(u)==int):
            res = np.zeros((dim,1))
        res[u] = 1
        return res
        assert(len(u)==dim)
        return np.array(u).reshape((-1,1))
    args = [(a,b,c,vec_u(u)) for (a,b,c,u) in args]

    HA_CBF = np.zeros((2*dim,2*dim))
    HA_CBF[dim:,dim:] = np.sum(
        a * u @ u.T for (a,b,c,u) in args)

    Hb_CBF = np.zeros(2*dim)
    Hb_CBF[dim:] = np.sum(b * u.reshape(-1) for (a,b,c,u) in args)

    Hc_CBF = np.sum(c for (a,b,c,u) in args)

    return HA_CBF,Hb_CBF,Hc_CBF


def CBF_GEN_degree0(dim, *args:(float,float,float,{int, np.array})):
    """
    Add constraint about position: u [a,0;a,0] u.T + [b;0].T u + c to the CBF
    arg dim: int     The total dimension of the state, CBF matrix is 2dim x 2dim
    arg *args: [(a,b,c,u)] x n 
    arg a,b,c:       The factor 
    arg u:         union{int, np.array([dim])} The dimension of the constraint
    """
    def vec_u(u):
        if(type(u)==int):
            res = np.zeros((dim,1))
        res[u] = 1
        return res
        assert(len(u)==dim)
        return np.array(u).reshape((-1,1))
    args = [(a,b,c,vec_u(u)) for (a,b,c,u) in args]

    HA_CBF = np.zeros((2*dim,2*dim))
    HA_CBF[:dim,:dim] = np.sum(
        a * u @ u.T for (a,b,c,u) in args)

    Hb_CBF = np.zeros(2*dim)
    Hb_CBF[:dim] = np.sum(b * u.reshape(-1) for (a,b,c,u) in args)

    Hc_CBF = np.sum(c for (a,b,c,u) in args)
    return HA_CBF,Hb_CBF,Hc_CBF