import torch
import torch.nn as nn
import numpy as np
import gym
from copy import deepcopy
from IEEE_Standard_Bus_Model import IEEE_Standard_Bus_Model
from PowerSystemController import IntegralController, PiecewiseController
from time_str import time_int

class Env_Closedloop_IEEE_Standard_Bus_Model(gym.Env):
    def __init__(self, nabla_F:callable, F_of_game:callable, omega_weight=200000, u_weight=2.0,
                  s_weight=2.0, s_dot_weight = 1.0,
                  u_dot_weight = 1.0, max_reward=2.0,
                  u_overline = None, u_underline = None,
                  N=68, money_loss_weight=1.0) -> None:
        self.power_model = IEEE_Standard_Bus_Model(N=N)
        self.dt = self.power_model.dt
        self.N = self.power_model.N_bus
        self.L = self.power_model.L
        self.L_B = self.power_model.L_B
        self.K_s = 100*np.eye(self.N, dtype=np.float32)
        self.K_lambda = 0.008*np.eye(self.N, dtype=np.float32)
        self.K_mu = 0.002*np.eye(self.N, dtype=np.float32)
        self.K_nu_minus = 0.02
        self.K_nu_plus = 0.02
        self.u_overline = 100000*np.ones([self.N, 1], dtype=np.float32) if(u_overline is None) else u_overline
        self.u_underline = -100000*np.ones([self.N, 1], dtype=np.float32) if(u_underline is None) else u_underline

        self.nabla_F = nabla_F
        self.nabla_F(None, N=self.N)
        self.F_of_game = F_of_game
        self.F_of_game(None, N=self.N)
        self.P_m = self.power_model.p

        self.omega_weight =omega_weight*np.ones([self.N, 1], dtype=np.float32)
        self.u_weight = u_weight*np.ones([self.N, 1], dtype=np.float32)
        self.s_weight = s_weight*np.ones([self.N, 1], dtype=np.float32)
        self.s_dot_weight = s_dot_weight*np.ones([self.N, 1], dtype=np.float32)
        self.u_dot_weight = u_dot_weight*np.ones([self.N, 1], dtype=np.float32)
        self.money_loss_weight = money_loss_weight*np.ones([self.N, 1], dtype=np.float32)
        self.max_reward = max_reward

        self.adjacency_matrix = self.power_model.AdjacencyMatrix
        self.s_s = np.zeros([self.power_model.N_bus, 1], dtype=np.float32)
        self.lambda_s = np.zeros([self.power_model.N_bus, 1], dtype=np.float32)
        self.mu_s = np.zeros([self.power_model.N_bus, 1], dtype=np.float32)
        self.nu_minus = np.zeros([self.N, 1], dtype=np.float32)
        self.nu_plus = np.zeros([self.N, 1], dtype=np.float32)
        self.u = np.zeros([self.N, 1], dtype=np.float32)
        self.u_dot = np.zeros([self.N, 1], dtype=np.float32)
        self.u_noise = np.zeros([self.N, 1], dtype=np.float32)

        self.neighbours = {}
        for i in range(self.power_model.N_bus):
            self.neighbours[i] = []
            for j in range(self.power_model.N_bus):
                if(self.adjacency_matrix[i][j]!=0):
                    self.neighbours[i].append(j)
        self._obs_num_local = 10
        # self._obs_num_local = 9
        self.obs_num_s = []
        for i in range(self.power_model.N_bus):
            self.obs_num_s.append(self._obs_num_local+3*len(self.neighbours[i]))
        self.action_num_s = [1]*self.N
        self.obs_s = []
        for i in range(self.N):
            self.obs_s.append(np.zeros([self.obs_num_s[i], 1], dtype=np.float32))
        self.init_state = 0

    def reset(self, update_p = False, p = None):
        if (update_p):
            if(p is not None):
                self.power_model.set_new_p(p)
            else:
                # choices = [-1, -0.75, 0.75, 1]
                choices = [1, -1]
                # p = self.power_model._p*(2*np.random.rand(self.N, 1)-1)
                p = self.power_model._p * (choices[self.init_state%len(choices)])
                self.power_model.set_new_p(p)
                self.init_state = self.init_state+1
        obs = self.power_model.reset()
        theta = obs[0:self.power_model.N_bus]
        omega = obs[self.power_model.N_bus:2*self.power_model.N_bus]

        self.s_s *= 0
        self.nu_minus *= 0
        self.nu_plus *= 0
        self.lambda_s *= 0
        self.lambda_s += 0.0
        self.mu_s *= 0
        self.u *= 0
        self.u_dot *= 0
        self.u_noise *= 0

        for i in range(self.N):
            self.obs_s[i][0:self._obs_num_local] = \
                [theta[i], omega[i], self.s_s[i], self.nu_minus[i], 
                 self.nu_plus[i], self.lambda_s[i], self.mu_s[i], self.u_dot[i], self.u[i], self.P_m[i]]
            # self.obs_s[i][0:self._obs_num_local] = \
            #     [theta[i], omega[i], self.s_s[i], self.nu_minus[i], 
            #      self.nu_plus[i], self.lambda_s[i], self.mu_s[i], self.u_dot[i], self.u[i]]
            for j, neighbour in enumerate(self.neighbours[i]):
                self.obs_s[i][self._obs_num_local+j:self._obs_num_local+(j+1)] = np.sin(theta[neighbour]-theta[i])*self.L_B[i][neighbour]
            for j, neighbour in enumerate(self.neighbours[i]):
                self.obs_s[i][self._obs_num_local+len(self.neighbours[i])+j:self._obs_num_local+len(self.neighbours[i])+(j+1)] = self.u_dot[neighbour]
            for j, neighbour in enumerate(self.neighbours[i]):
                self.obs_s[i][self._obs_num_local+len(self.neighbours[i])+j:self._obs_num_local+len(self.neighbours[i])+(j+1)] = self.P_m[neighbour]


        self.step(self.u)
        u = 1*self.s_s[:]
        self.step(u)
        return deepcopy(self.obs_s)

    def step(self, u, u_noise=None):
        obs = self.power_model.step(u)
        theta = obs[0:self.power_model.N_bus]
        omega = obs[self.power_model.N_bus:2*self.power_model.N_bus]
        # improved Euler method
        # \dot{y} = f(t, y)
        # z_{n+1} = y_n+hf(t_n, y_n)
        # y_{n+1} = y_n+\frac{h}{2} (f(t_n, y_n)+f(t_{n+1}, z_{n+1}))
        # 2 order accuracy. (While Euler 1 order)
        dot_s_1 = -self.K_s@(self.nabla_F(u)+omega+self.lambda_s-self.nu_minus+self.nu_plus)
        dot_nu_minus_1 = self.K_nu_minus*self.projectionOnPositive(self.u_underline-u, self.nu_minus)
        dot_nu_plus_1 = self.K_nu_plus*self.projectionOnPositive(u-self.u_overline, self.nu_plus)
        dot_lambda_1 = self.K_lambda@(-self.L@self.lambda_s-self.L@self.mu_s+self.P_m+u)
        dot_mu_1 = self.K_mu@(self.L@self.lambda_s)

        s_2 = self.s_s + self.dt*dot_s_1
        nu_minus_2 = self.nu_minus + self.dt*dot_nu_minus_1
        nu_plus_2 = self.nu_plus + self.dt*dot_nu_plus_1
        lambda_2 = self.lambda_s + self.dt*dot_lambda_1
        mu_2 = self.mu_s + self.dt*dot_mu_1

        dot_s_2 = -self.K_s@(self.nabla_F(u)+omega+lambda_2-nu_minus_2+nu_plus_2)
        dot_nu_minus_2 = self.K_nu_minus*self.projectionOnPositive(self.u_underline-u, nu_minus_2)
        dot_nu_plus_2 = self.K_nu_plus*self.projectionOnPositive(u-self.u_overline, nu_plus_2)
        dot_lambda_2 = self.K_lambda@(-self.L@lambda_2-self.L@mu_2+self.P_m+u)
        dot_mu_2 = self.K_mu@(self.L@lambda_2)
        
        if(u_noise is None):
            dot_u = (u-(self.u-self.u_noise))/self.dt
            self.u_noise[:] = 0
        else:
            dot_u = ((u-u_noise)-(self.u-self.u_noise))/self.dt
            self.u_noise[:] = u_noise

        self.u[:] = u
        self.s_s += self.dt/2*(dot_s_1+dot_s_2)
        self.nu_minus += self.dt/2*(dot_nu_minus_1+dot_nu_minus_2)
        self.nu_plus += self.dt/2*(dot_nu_plus_1+dot_nu_plus_2)
        self.lambda_s += self.dt/2*(dot_lambda_1+dot_lambda_2)
        self.mu_s += self.dt/2*(dot_mu_1+dot_mu_2)
        
        self.u_dot[:] = dot_u

        for i in range(self.N):
            self.obs_s[i][0:self._obs_num_local] = \
                [theta[i], omega[i], self.s_s[i], self.nu_minus[i], 
                 self.nu_plus[i], self.lambda_s[i], self.mu_s[i], self.u_dot[i], self.u[i], self.P_m[i]]
            # self.obs_s[i][0:self._obs_num_local] = \
            #     [theta[i], omega[i], self.s_s[i], self.nu_minus[i], 
            #      self.nu_plus[i], self.lambda_s[i], self.mu_s[i], self.u_dot[i], self.u[i]]
            for j, neighbour in enumerate(self.neighbours[i]):
                self.obs_s[i][self._obs_num_local+j:self._obs_num_local+(j+1)] =np.sin(theta[neighbour]-theta[i])*self.L_B[i][neighbour]
            for j, neighbour in enumerate(self.neighbours[i]):
                self.obs_s[i][self._obs_num_local+len(self.neighbours[i])+j:self._obs_num_local+len(self.neighbours[i])+(j+1)] = self.u_dot[neighbour]
            for j, neighbour in enumerate(self.neighbours[i]):
                self.obs_s[i][self._obs_num_local+len(self.neighbours[i])+j:self._obs_num_local+len(self.neighbours[i])+(j+1)] = self.P_m[neighbour]

        dot_s = (dot_s_1+dot_s_2)/2

        u_loss = self.u_weight*abs(u)
        omega_loss = self.omega_weight*abs(omega)
        s_loss = self.s_weight*abs(self.s_s)
        dot_s_loss = self.s_dot_weight*abs(dot_s)
        dot_u_loss = self.u_dot_weight*abs(dot_u)
        money_loss = self.money_loss_weight*self.F_of_game(u)
        loss = u_loss**2+omega_loss**2+s_loss**2+dot_s_loss**2+dot_u_loss**2+money_loss

        # reward_s = np.fmax(self.max_reward - loss, 0)
        reward_s = self.max_reward-loss
        if(np.max(np.abs(omega[:])) > 0.1):
            done = 1.0
            print("Power Model is in danger, termited {:10.0f}".format(time_int()))
        elif(np.max(np.abs(omega[:])) < 0.00005 and np.abs(np.sum(self.P_m-u)) < 0.3):
            done = 0.0
        else:
            done = 0.0
        info = {}
        reward_s = reward_s*self.dt
        return deepcopy(self.obs_s), reward_s, done, info

    def projectionOnPositive(self, x, a):
        """
        [x]_a^+
        """
        y = np.logical_or(x>0, a>0)*x
        return y
