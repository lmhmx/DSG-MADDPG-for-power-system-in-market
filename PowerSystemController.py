import numpy as np
import matplotlib.pyplot as plt
import torch

class PowerSystemControllerBase:
    def __init__(self) -> None:
        pass
    def output(self, state) -> None:
        pass


class IntegralController(PowerSystemControllerBase):
    def __init__(self, dt, N_bus) -> None:
        self.u = 0
        self.dt = dt
        self.K = 100
        self.N_bus = N_bus

    def output(self, state) -> None:
        omega = state[self.N_bus:self.N_bus*2]
        self.u = self.u-self.dt*self.K*omega
        return self.u


class PiecewiseController(PowerSystemControllerBase):
    def __init__(self, N, d, L, dt, nabla_F:callable, P_m, out_scale:callable=np.copy,
                 u_overline = None, u_underline = None) -> None:
        """
        N: the number of agents
        d: the number of pieces
        L: Laplacian matrix of the graph
        """
        self.N = N
        self.d = d
        self.L = L
        self.dt = dt
        self.K_s = 100*np.eye(self.N)
        self.out_scale = out_scale
        self.K_lambda = 0.008*np.eye(self.N)
        self.K_mu = 0.002*np.eye(self.N)

        self.initPiecewiseParam()
        self.u_overline = 100000*np.ones([N, 1]) if(u_overline is None) else u_overline
        self.u_underline = -100000*np.ones([N, 1]) if(u_underline is None) else u_underline
        self.nu_minus = np.zeros([N, 1])
        self.nu_plus = np.zeros([N, 1])
        self.lambda_ = np.zeros([N, 1])
        self.mu_ = np.zeros([N, 1])
        self.s = np.zeros([N, 1])
        self.u = np.zeros([N, 1])
        self.dot_s = np.zeros([N, 1])
        self.dot_nu_minus = np.zeros([N, 1])
        self.dot_nu_plus = np.zeros([N, 1])
        self.dot_lambda = np.zeros([N, 1])
        self.dot_mu = np.zeros([N, 1])
        self.dot_u = np.zeros([N, 1])
        self.nabla_F = nabla_F
        self.nabla_F(None, N=self.N)
        self.P_m = P_m

    def initPiecewiseParam(self):
        a = 1/self.d
        a = np.sqrt(a)
        self.kappa_plus = np.ones([self.N, self.d])
        self.kappa_minus = np.ones([self.N, self.d])
        
        self.beta_plus = a*np.ones([self.N, self.d-1])
        self.beta_minus = a*np.ones([self.N, self.d-1])

        self.k_plus = self.kappa_plus*self.kappa_plus
        self.k_minus = self.kappa_minus*self.kappa_minus
        
        self.b_plus =  np.zeros([self.N, self.d])
        for i in range(1, self.d):
            self.b_plus[:,i] = np.sum(self.beta_plus[:,0:i]*self.beta_plus[:, 0:i], axis=1)
        self.b_minus = np.zeros([self.N, self.d])
        for i in range(1, self.d):
            self.b_minus[:,i] = -np.sum(self.beta_minus[:,0:i]*self.beta_minus[:, 0:i], axis=1)

    # def initPiecewiseParam(self):
    #     np.random.seed(1)
    #     # self.mu_plus = np.random.randn(self.N, self.d)
    #     # self.chi_plus = np.random.rand(self.N, self.d-1)
    #     # self.mu_minus = np.random.randn(self.N, self.d)
    #     # self.chi_minus = np.random.randn(self.N, self.d)

    #     self.mu_plus = np.ones([self.N, self.d])
    #     self.chi_plus = np.ones([self.N, self.d-1])

    #     self.mu_minus = np.ones([self.N, self.d])
    #     self.chi_minus = np.ones([self.N, self.d-1])

    #     self.k_plus = self.mu_plus*self.mu_plus
    #     self.k_plus[:,1:] = self.k_plus[:,1:] - self.mu_plus[:,0:-1]*self.mu_plus[:,0:-1]

    #     self.b_plus = np.zeros([self.N, self.d])
    #     for i in range(1, self.d):
    #         self.b_plus[:,i] = np.sum(self.chi_plus[:,0:i]*self.chi_plus[:, 0:i], axis=1)

        
    #     self.k_minus = -self.mu_minus*self.mu_minus
    #     self.k_minus[:, 1:] = self.k_minus[:, 1:] + self.mu_minus[:, 0:-1]*self.mu_minus[:, 0:-1]
        
    #     self.b_minus = np.zeros([self.N, self.d])
    #     for i in range(1, self.d):
    #         self.b_minus[:,i] = -np.sum(self.chi_minus[:,0:i]*self.chi_minus[:, 0:i], axis=1)
    def output(self, state):
        omega = state[self.N:self.N*2]
        u = self.monotoneFunction(self.s)
        self.dot_s[:] = -self.K_s@(self.nabla_F(u)+omega+self.lambda_-self.nu_minus+self.nu_plus)
        self.dot_nu_minus[:] = 0.02*self.projectionOnPositive(self.u_underline-u, self.nu_minus)
        self.dot_nu_plus[:] = 0.02*self.projectionOnPositive(u-self.u_overline, self.nu_plus)
        self.dot_lambda[:] = self.K_lambda@(-self.L@self.lambda_-self.L@self.mu_+self.P_m+u)
        self.dot_mu[:] = self.K_mu@(self.L@self.lambda_)
        
        self.s += self.dt*self.dot_s
        self.nu_minus += self.dt*self.dot_nu_minus
        self.nu_plus += self.dt*self.dot_nu_plus
        self.lambda_ += self.dt*self.dot_lambda
        self.mu_ += self.dt*self.dot_mu
        u = self.monotoneFunction(self.s)
        
        self.dot_u = (u-self.u)/self.dt
        self.u[:] = u

        return u
    def monotoneFunction(self, s):
        k_plus = self.k_plus
        k_minus = self.k_minus
        b_plus = self.b_plus
        b_minus = self.b_minus
        u_plus_1 = np.sum(k_plus*self.relu(s-b_plus), axis = 1, keepdims=True)
        u_plus_2 = -np.sum(k_plus[:, 0:self.d-1]*self.relu(s-b_plus[:,1:]),axis = 1, keepdims=True)
            
        u_minus_1 = -np.sum(k_minus*self.relu(b_minus-s), axis = 1, keepdims=True)
        u_minus_2 = np.sum(k_minus[:, 0:self.d-1]*self.relu(b_minus[:,1:]-s), axis = 1, keepdims=True)
        action = u_plus_1 + u_plus_2 + u_minus_1 + u_minus_2

        return self.out_scale(action)
    
        # u_plus=np.sum(self.k_plus*self.relu(s-self.b_plus), axis=1, keepdims=True)
        # u_minus=np.sum(self.k_minus*self.relu(self.b_minus-s), axis=1, keepdims=True)
        # u=u_plus+u_minus
        # return self.out_scale(u)

    
    def projectionOnPositive(self, x, a):
        """
        [x]_a^+
        """
        y = np.logical_or(x>0, a>0)*x
        return y
    def relu(self, s):
        return np.fmax(s, 0)
    def plotMonotoneFunction(self):
        range_plus = np.max(self.b_plus)*1.5
        range_minus = np.min(self.b_minus)*1.5
        f_s = np.zeros([100,self.N])
        plot_range = np.linspace(range_minus, range_plus, 100)
        for i,t in enumerate(plot_range):
            s = np.zeros([self.N,1])+t
            f_s[i,:] = self.monotoneFunction(s).flatten()
        plt.figure()
        plt.plot(plot_range, f_s)
        plt.show()


if(__name__=="__main__"):
    from Env_IEEE_Standard_Bus_Model import Env_IEEE_Standard_Bus_Model
    from nabla_F_of_game import nabla_F_of_game
    env = Env_IEEE_Standard_Bus_Model()
    controller = PiecewiseController(N=env.N_bus, d=15, L=env.power_model.L, 
                                 dt=env.power_model.dt, 
                                 nabla_F=nabla_F_of_game, 
                                 P_m=env.power_model.p)
    controller.plotMonotoneFunction()
    