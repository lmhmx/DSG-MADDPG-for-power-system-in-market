import numpy as np
from typing import Optional, Tuple
from load_ieee_parameters import load_ieee_parameters
from scipy.integrate import odeint
from time_str import time_int
class IEEE_Standard_Bus_Model:
    def __init__(self, N=68) -> None:
        super().__init__()
        power_sys_param = load_ieee_parameters(N=N)
        self.M_g = np.array(power_sys_param["M_g"], dtype=np.float32)
        self._M_g_inv = np.linalg.inv(self.M_g)
        self.f_0 = np.array(power_sys_param["f_0"], dtype=np.float32)
        self.D = np.array(power_sys_param["D"], dtype=np.float32)
        self.B = np.array(power_sys_param["B"], dtype=np.float32)
        self.Gamma_g = np.array(power_sys_param["Gamma_g"], dtype=np.float32)
        self.Gamma_l = np.array(power_sys_param["Gamma_l"], dtype=np.float32)
        self.N_bus = int(power_sys_param["N_bus"][0][0])
        self.N_g = int(power_sys_param["N_g"][0][0])
        self.N_l = int(power_sys_param["N_l"][0][0])
        self.N_line = int(power_sys_param["N_line"][0][0])
        self.R = np.array(power_sys_param["R"], dtype=np.float32)
        # # [X] (This is wrong)Laplacian matrix for lambda and mu. It is calculated by weight of D
        # # The original controller is not good. Using the following controller
        # # self.L = self.generateLaplacianMatrixByNodeWeight(self.R, np.diag(self.D))
        # self.L = self.R@self.R.T
        # Wrong again
        # Laplacian matrix for lambda and mu. It is calculated by weight of D
        self.L = self.generateLaplacianMatrixByNodeWeight(self.R, 2*0.325*1.5*40*np.diag(self.D))
        # Laplacian matrix by RBR^T

        # Laplacian matrix by RBR^T
        self.L_B = self.R@self.B@self.R.T
        # Adjacency Matrix of the graph
        self.AdjacencyMatrix = self.generateAdjacencyMatrix(self.R)

        self.D_g = self.Gamma_g@self.D@self.Gamma_g.T
        self.D_l = self.Gamma_l@self.D@self.Gamma_l.T
        self._D_l_inv = np.linalg.inv(self.D_l)
        self.omega_g = np.zeros([self.N_g,1], dtype=np.float32)
        self.omega_l = np.zeros([self.N_l,1], dtype=np.float32)
        self.omega = self.Gamma_g.T@self.omega_g+self.Gamma_l.T@self.omega_l
        self.theta = np.zeros([self.N_bus,1], dtype=np.float32)
        self.dot_omega = np.zeros([self.N_bus, 1], dtype=np.float32)
        self.dot_theta = np.zeros([self.N_bus, 1], dtype=np.float32)
        self.p = np.zeros([self.N_bus,1], dtype=np.float32)
        if(self.N_bus == 68):
            self.p[3] =-5
            self.p[7] =-5
            self.p[19] =-5
            self.p[36] =-5
            self.p[41] =-5
            self.p[51] =-10
        elif(self.N_bus == 3):
            self.p[0] = -0.9
            self.p[1] = -0.6
            self.p[2] = -0.
        else:
            raise NotImplemented("N_bus error")
        self._p = self.p.copy()
        self.u = np.zeros([self.N_bus,1], dtype=np.float32)
        self.obs = np.zeros([2*self.N_bus,1], dtype=np.float32)
        self.dt_num_rate = 50
        self._dt = 0.0001
        self.dt = self.dt_num_rate*self._dt
        # self._dt = self.dt/self.dt_num_rate
        self.use_ode_int = False

    def generateAdjacencyMatrix(self, R):
        n = R.shape[0]
        m = R.shape[1]
        A = np.zeros([n, n])
        for i in range(m):
            n1 = -1
            n2 = -1
            for j in range(n):
                if(R[j][i]!=0 and n1==-1):
                    n1 = j
                elif(R[j][i]!=0 and n1!=-1):
                    n2 = j
            A[n1][n2] = 1
            A[n2][n1] = 1
        return A

    def generateLaplacianMatrixByNodeWeight(self, R:np.ndarray, w:np.ndarray):
        """
        R: incidence matrix
        w: node weight
        return:
            L: Laplacian matrix weighted by w
        """
        w = w.flatten()
        n = R.shape[0]
        m = R.shape[1]
        B = np.zeros([m, m], dtype=np.float32)
        for i in range(m):
            weight_i = 0
            for j in range(n):
                if(R[j][i]!=0):
                    weight_i += w[j]
            B[i][i] = weight_i
        L = R@B@R.T
        return L
    def ode_int_f(self, y, t, u_l, u_g, p_l, p_g):
        # u_l = self.Gamma_l@self.u
        # u_g = self.Gamma_g@self.u
        # p_l = self.Gamma_l@self.p
        # p_g = self.Gamma_g@self.p

        theta = np.reshape(y[0:self.N_bus], [-1, 1])
        omega_g = np.reshape(y[self.N_bus:self.N_bus+self.N_g], [-1, 1])
        omega_l = self._D_l_inv@(p_l+u_l-self.Gamma_l@self.R@self.B@np.sin(self.R.T@theta))
        
        dot_theta = 2*np.pi*self.f_0*(self.Gamma_g.T@omega_g+self.Gamma_l.T@omega_l)
        dot_omega_g = self._M_g_inv@(p_g+u_g-self.D_g@omega_g-self.Gamma_g@self.R@self.B@np.sin(self.R.T@theta))
        dot_y = np.vstack([dot_theta, dot_omega_g])
        return dot_y.flatten()
    def step(self, action:np.ndarray) -> np.ndarray:
        self.u[:] = action
        u_l = self.Gamma_l@self.u
        u_g = self.Gamma_g@self.u
        p_l = self.Gamma_l@self.p
        p_g = self.Gamma_g@self.p

        if(self.use_ode_int):
            tmp = odeint(self.ode_int_f, np.vstack([self.theta, self.omega_g]).flatten(),
                          [0, self.dt],
                          args=(u_l, u_g, p_l, p_g))
            y_after_dt = np.reshape(tmp[1], [-1,1])
            self.theta[:] = y_after_dt[0:self.N_bus]
            self.omega_g[:] = y_after_dt[self.N_bus:self.N_bus+self.N_g]
            self.omega_l[:] = self._D_l_inv@(p_l+u_l-self.Gamma_l@self.R@self.B@np.sin(self.R.T@self.theta))
            self.omega[:] = self.Gamma_g.T@self.omega_g+self.Gamma_l.T@self.omega_l
        else:
            for i in range(self.dt_num_rate):
                dot_theta = 2*np.pi*self.f_0*self.omega

                dot_omega_g = self._M_g_inv@(p_g+u_g-self.D_g@self.omega_g-self.Gamma_g@self.R@self.B@np.sin(self.R.T@self.theta))

                self.omega_g[:] = self.omega_g+dot_omega_g*self._dt
                self.omega_l[:] = self._D_l_inv@(p_l+u_l-self.Gamma_l@self.R@self.B@np.sin(self.R.T@self.theta))

                self.omega[:] = self.Gamma_g.T@self.omega_g+self.Gamma_l.T@self.omega_l
                self.theta[:] = self.theta+dot_theta*self._dt
        self.dot_theta[:] = (self.theta - self.obs[0:self.N_bus])/self.dt
        self.dot_omega[:] = (self.obs[self.N_bus:2*self.N_bus]-self.omega)/self.dt
        self.obs[0:self.N_bus]=self.theta
        self.obs[self.N_bus:2*self.N_bus]=self.omega
        return self.obs.copy()
    def reset(self):
        self.omega[:] = 0*self.omega
        self.theta[:] = 0*self.theta
        self.dot_omega[:] = 0*self.dot_omega
        self.dot_theta[:] = 0*self.dot_theta
        self.omega_g[:] = 0*self.omega_g
        self.omega_l[:] = 0*self.omega_l
        self.obs[:] = 0*self.obs
        self.u[:] = 0*self.u
        self.step(self.u)
        return self.obs.copy()
    def set_new_p(self, p):
        self.p[:] = p[:]

if(__name__ == "__main__"):
    m = IEEE_Standard_Bus_Model()
    max_time = 1
    it_num = int(max_time/m.dt)
    print(time_int())
    for i in range(it_num):
        m.step(np.zeros([m.N_bus,1], dtype=np.float32))
    print(time_int())

