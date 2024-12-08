import numpy as np

def nabla_F_of_game(u, N = None):
    if(N is not None):
        if(N==68):
            F_gain = 0.008
            # nabla_F_of_game.alpha = F_gain*1.0*np.eye(N, dtype=np.float32)
            nabla_F_of_game.alpha = F_gain*1.0* np.diag(np.hstack([0.75*np.ones([52]), np.ones([16])])).astype(np.float32)
            nabla_F_of_game.a = F_gain*0.2*0.05
            nabla_F_of_game.b = F_gain*0.2*0

            nabla_F_of_game.u_param = np.zeros([N, N], dtype=np.float32)
            nabla_F_of_game.u_param += nabla_F_of_game.alpha
            nabla_F_of_game.u_param += nabla_F_of_game.a*np.eye(N, dtype=np.float32)
            nabla_F_of_game.u_param += nabla_F_of_game.a*np.ones([N,N], dtype=np.float32)
            nabla_F_of_game.bias = nabla_F_of_game.b*np.ones([N,1], dtype=np.float32)
        elif(N==3):
            F_gain = 0.008
            # nabla_F_of_game.alpha = F_gain*1.0*np.eye(N, dtype=np.float32)
            nabla_F_of_game.alpha = F_gain*1.0* np.diag(np.hstack([1.0*np.ones([1]), 1.0*np.ones([1]), np.ones([1])])).astype(np.float32)
            nabla_F_of_game.a = F_gain*0.2*0.05
            nabla_F_of_game.b = F_gain*0.2*0

            nabla_F_of_game.u_param = np.zeros([N, N], dtype=np.float32)
            nabla_F_of_game.u_param += nabla_F_of_game.alpha
            nabla_F_of_game.u_param += nabla_F_of_game.a*np.eye(N, dtype=np.float32)
            nabla_F_of_game.u_param += nabla_F_of_game.a*np.ones([N,N], dtype=np.float32)
            nabla_F_of_game.bias = nabla_F_of_game.b*np.ones([N,1], dtype=np.float32)
        return
    res = nabla_F_of_game.u_param@u+nabla_F_of_game.bias
    return res

def F_of_game(u:np.ndarray, N = None, is_batch = False):
    if(N is not None):
        if(N==68):
            # nabla_F_of_game.alpha = 0.02*1.0*np.eye(N)
            # nabla_F_of_game.a = 0.02*0.2*0.05
            # nabla_F_of_game.b = 0.02*0.2
            F_gain = 0.008
            # nabla_F_of_game.alpha = F_gain*1.0*np.eye(N, dtype=np.float32)
            F_of_game.alpha = F_gain*1.0* np.diag(np.hstack([0.75*np.ones([52]), np.ones([16])])).astype(np.float32)
            F_of_game.a = F_gain*0.2*0.05
            F_of_game.b = F_gain*0.2*0

            # F_of_game.u_param = np.zeros([N, N], dtype=np.float32)
            # F_of_game.u_param += nabla_F_of_game.alpha
            # F_of_game.u_param += nabla_F_of_game.a*np.eye(N, dtype=np.float32)
            # F_of_game.u_param += nabla_F_of_game.a*np.ones([N,N], dtype=np.float32)
            # F_of_game.bias = nabla_F_of_game.b*np.ones([N,1], dtype=np.float32)
        elif(N==3):
            # nabla_F_of_game.alpha = 0.02*1.0*np.eye(N)
            # nabla_F_of_game.a = 0.02*0.2*0.05
            # nabla_F_of_game.b = 0.02*0.2
            F_gain = 0.008
            # nabla_F_of_game.alpha = F_gain*1.0*np.eye(N, dtype=np.float32)
            F_of_game.alpha = F_gain*1.0* np.diag(np.hstack([1.0*np.ones([1]), 1.0*np.ones([1]), np.ones([1])])).astype(np.float32)
            F_of_game.a = F_gain*0.2*0.05
            F_of_game.b = F_gain*0.2*0
        return
    if(is_batch): # u(m,N)
        res = 0.5*u*(u@F_of_game.alpha)+u*(F_of_game.b+F_of_game.a*u.sum(axis=1,keepdims=True))
    else: # u(N,1)
        res = 0.5*u*(F_of_game.alpha@u)+u*(F_of_game.b+F_of_game.a*u.sum())
    return res

