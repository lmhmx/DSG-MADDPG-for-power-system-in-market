import numpy as np
import torch
import torch.nn as nn
from Policy_Controller_base import Policy_Controller_base

class Policy_Piecewise_Controller(Policy_Controller_base):
    def __init__(self, N, d=16, device="cpu") -> None:
        super(Policy_Piecewise_Controller, self).__init__()
        self.N = N
        self.d = d
        self.device = device
        self.initPiecewiseParam()
        self.out_layer = nn.Identity()
        self.relu_layer = nn.ReLU()
    def initPiecewiseParam(self):
        a = 0.4/self.d
        a = np.sqrt(a)
        self.kappa_plus = nn.parameter.Parameter(data=torch.ones([self.N, self.d], device=self.device))
        self.kappa_minus = nn.parameter.Parameter(data=torch.ones([self.N, self.d], device=self.device))
        
        self.beta_plus = nn.parameter.Parameter(data=a*torch.ones([self.N, self.d-1], device=self.device))
        self.beta_minus = nn.parameter.Parameter(data=a*torch.ones([self.N, self.d-1], device=self.device))

    @property
    def K(self):
        return 0.5*(self.k_plus.mean(dim=1, keepdim=True)+self.k_minus.mean(dim=1, keepdim=True))

    @property
    def k_plus(self):
        self.k_plus_ = self.kappa_plus*self.kappa_plus
        return self.k_plus_
    @property
    def k_minus(self):
        self.k_minus_ = self.kappa_minus*self.kappa_minus
        return self.k_minus_
    @property
    def b_plus(self):
        self.b_plus_ =  torch.zeros([self.N, self.d], device=self.device)
        for i in range(1, self.d):
            self.b_plus_[:,i] = torch.sum(self.beta_plus[:,0:i]*self.beta_plus[:, 0:i], axis=1)
        return self.b_plus_
    @property
    def b_minus(self):
        self.b_minus_ = torch.zeros([self.N, self.d], device=self.device)
        for i in range(1, self.d):
            self.b_minus_[:,i] = -torch.sum(self.beta_minus[:,0:i]*self.beta_minus[:, 0:i], axis=1)
        return self.b_minus_
    def forward(self, obs_s:list[np.ndarray], is_batch = False):
        """
        obs_s: 里面的每个向量的定义方式
            * 若is_batch 为真，则向量为 batch_size * n
            * 若is_batch 为假，则向量为 n*1
            注意：保留这种做法属于迫不得已，在构建前期模型时没考虑到batch的问题，
                导致用n*1的矩阵定义列向量的方法与torch和numpy的写法造成了不匹配，
                重新适配比较复杂，因此直接在后面加了一个is_batch参数，
                以后构建新程序时，记住要把向量定义成(n, )的shape，不要搞成(n,1)
        """
        k_plus = self.k_plus
        k_minus = self.k_minus
        b_plus = self.b_plus
        b_minus = self.b_minus
        if(is_batch):
            batch_size = obs_s[0].shape[0]
            s_s = torch.zeros([batch_size, self.N, 1], device=self.device)
            for i in range(self.N):
                s_s[:, i, 0] = obs_s[i][:, 2]

            u_plus_1 = torch.sum(k_plus*self.relu_layer(s_s-b_plus), axis = 2)
            u_plus_2 = -torch.sum(k_plus[:, 0:self.d-1]*self.relu_layer(s_s-b_plus[:,1:]), axis=2)
            
            u_minus_1 = -torch.sum(k_minus*self.relu_layer(b_minus-s_s), axis = 2)
            u_minus_2 = torch.sum(k_minus[:, 0:self.d-1]*self.relu_layer(b_minus[:,1:]-s_s), axis = 2)
            action = u_plus_1 + u_plus_2 + u_minus_1 + u_minus_2
        else:
            s_s = torch.zeros([self.N, 1], device=self.device)
            for i in range(self.N):
                s_s[i, :] = torch.as_tensor(obs_s[i][2], device=self.device)
            # u_plus = torch.sum(self.k_plus*self.relu_layer(s_s-self.b_plus), axis = 1, keepdim=True)

            # u_minus = torch.sum(self.k_minus*self.relu_layer(self.b_minus-s_s), axis = 1, keepdim=True)
            
            u_plus_1 = torch.sum(k_plus*self.relu_layer(s_s-b_plus), axis = 1, keepdim=True)
            u_plus_2 = -torch.sum(k_plus[:, 0:self.d-1]*self.relu_layer(s_s-b_plus[:,1:]),axis = 1, keepdim=True)
            
            u_minus_1 = -torch.sum(k_minus*self.relu_layer(b_minus-s_s), axis = 1, keepdim=True)
            u_minus_2 = torch.sum(k_minus[:, 0:self.d-1]*self.relu_layer(b_minus[:,1:]-s_s), axis = 1, keepdim=True)
            action = u_plus_1 + u_plus_2 + u_minus_1 + u_minus_2
        return 1.0*self.out_layer(action)



class Policy_Piecewise_Controller_jiang(Policy_Controller_base):
    def __init__(self, N, d=16, device="cpu") -> None:
        super(Policy_Piecewise_Controller_jiang, self).__init__()
        self.N = N
        self.d = d
        self.device = device
        self.initPiecewiseParam()
        self.out_layer = nn.Identity()
        self.relu_layer = nn.ReLU()
    def initPiecewiseParam(self):
        a = 1/self.d
        a = np.sqrt(a)
        self.mu_plus = nn.parameter.Parameter(data=torch.ones([self.N, self.d], device=self.device))
        self.chi_plus = nn.parameter.Parameter(data=a*torch.ones([self.N, self.d-1], device=self.device))

        self.mu_minus = nn.parameter.Parameter(data=torch.ones([self.N, self.d], device=self.device))
        self.chi_minus = nn.parameter.Parameter(data=a*torch.ones([self.N, self.d-1], device=self.device))

    @property
    def K(self):
        return 0.5*(self.mu_plus.mean(dim=1, keepdim=True)+self.mu_minus.mean(dim=1, keepdim=True))

    @property
    def k_plus(self):
        self.k_plus_ = self.mu_plus*self.mu_plus
        self.k_plus_[:,1:] = self.k_plus_[:,1:] - self.mu_plus[:,0:-1]*self.mu_plus[:,0:-1]
        return self.k_plus_
    @property
    def k_minus(self):
        self.k_minus_ = -self.mu_minus*self.mu_minus
        self.k_minus_[:, 1:] = self.k_minus_[:, 1:] + self.mu_minus[:, 0:-1]*self.mu_minus[:, 0:-1]
        return self.k_minus_
    @property
    def b_plus(self):
        self.b_plus_ =  torch.zeros([self.N, self.d], device=self.device)
        for i in range(1, self.d):
            self.b_plus_[:,i] = torch.sum(self.chi_plus[:,0:i]*self.chi_plus[:, 0:i], axis=1)
        return self.b_plus_
    @property
    def b_minus(self):
        self.b_minus_ = torch.zeros([self.N, self.d], device=self.device)
        for i in range(1, self.d):
            self.b_minus_[:,i] = -torch.sum(self.chi_minus[:,0:i]*self.chi_minus[:, 0:i], axis=1)
        return self.b_minus_
    def forward(self, obs_s:list[np.ndarray], is_batch = False):
        """
        obs_s: 里面的每个向量的定义方式
            * 若is_batch 为真，则向量为 batch_size * n
            * 若is_batch 为假，则向量为 n*1
            注意：保留这种做法属于迫不得已，在构建前期模型时没考虑到batch的问题，
                导致用n*1的矩阵定义列向量的方法与torch和numpy的写法造成了不匹配，
                重新适配比较复杂，因此直接在后面加了一个is_batch参数，
                以后构建新程序时，记住要把向量定义成(n, )的shape，不要搞成(n,1)
        """
        if(is_batch):
            batch_size = obs_s[0].shape[0]
            s_s = torch.zeros([batch_size, self.N, 1], device=self.device)
            for i in range(self.N):
                s_s[:, i, 0] = obs_s[i][:, 2]
            u_plus = torch.sum(self.k_plus*self.relu_layer(s_s-self.b_plus), axis = 2)
            u_minus = torch.sum(self.k_minus*self.relu_layer(self.b_minus-s_s), axis = 2)
            action = u_plus+u_minus
        else:
            s_s = torch.zeros([self.N, 1], device=self.device)
            for i in range(self.N):
                s_s[i, :] = torch.as_tensor(obs_s[i][2], device=self.device)
            u_plus = torch.sum(self.k_plus*self.relu_layer(s_s-self.b_plus), axis = 1, keepdim=True)
            u_minus = torch.sum(self.k_minus*self.relu_layer(self.b_minus-s_s), axis = 1, keepdim=True)
            action = u_plus+u_minus
        return 1.0*self.out_layer(action)

