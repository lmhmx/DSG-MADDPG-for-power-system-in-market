import numpy as np
import torch
import torch.nn as nn
from Policy_Controller_base import Policy_Controller_base
class Policy_Linear_Controller(Policy_Controller_base):
    def __init__(self, N, device="cpu") -> None:
        super(Policy_Linear_Controller, self).__init__()
        self.device = device
        self.K_ = nn.parameter.Parameter(data=torch.ones([N, 1], device=self.device))
        self.N = N
        self.out_layer = nn.Identity()
    @property
    def K(self):
        return self.K_
    def huber(self, k:torch.Tensor, sigma = 1.0):
        huber_k = (k>sigma)*(sigma*k.abs()-0.5*sigma*sigma)+(k<=sigma)*(0.5*(k*k))
        return 2*huber_k
    @property
    def K_p(self):
        # return self.K.abs()
        return self.K*self.K
        # return self.huber(self.K)
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
            s_s = torch.zeros([batch_size, self.N], device=self.device)
            for i in range(self.N):
                s_s[:, i] = obs_s[i][:, 2]
            action = (self.K_p*s_s.T).T
        else:
            s_s = torch.zeros([self.N, 1], device=self.device)
            for i in range(self.N):
                s_s[i, :] = torch.tensor(obs_s[i][2], device=self.device)
            action = self.K_p*s_s
        return 1.0*self.out_layer(action)

