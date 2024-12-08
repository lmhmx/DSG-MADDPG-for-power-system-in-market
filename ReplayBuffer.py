import numpy as np
import torch
from nabla_F_of_game import F_of_game
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_nums, act_nums, size, device = "cpu", load_path = None,
                 re_compute_reward=False,
                   u_weight=1.0, s_weight =0,omega_weight=0,
                   dot_s_weight=0, dot_u_weight=0, max_reward=1.0,dt=0.005,
                    money_loss_weight=0.0 ):
        """
        act_nums: only 1 act for one agent
        """
        self.N = len(obs_nums)
        self.device = device
        if(load_path==None):
            self.obs_s_buf = []
            for i in range(len(obs_nums)):
                self.obs_s_buf.append(np.zeros([size, obs_nums[i]], dtype=np.float32))
            self.obs2_s_buf = []
            for i in range(len(obs_nums)):
                self.obs2_s_buf.append(np.zeros([size, obs_nums[i]], dtype=np.float32))
            self.act_buf = np.zeros([size, len(act_nums)], dtype=np.float32)
            self.rew_buf = np.zeros([size, len(act_nums)], dtype=np.float32)
            self.done_buf = np.zeros([size, 1], dtype=np.float32)

            self.ptr, self.size, self.max_size = 0, 0, size
        else:
            self.load(load_path, re_compute_reward,
                   u_weight, s_weight,omega_weight,
                   dot_s_weight, dot_u_weight, max_reward, dt, money_loss_weight=money_loss_weight)

    def store(self, obs, act, rew, next_obs, done):
        for i in range(self.N):
            self.obs_s_buf[i][self.ptr] = obs[i].flatten()
            self.obs2_s_buf[i][self.ptr] = next_obs[i].flatten()
        self.act_buf[self.ptr] = act.flatten()
        self.rew_buf[self.ptr] = rew.flatten()
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, sample_all = False, sample_last_n = False):
        if(sample_all):
            idxs = np.arange(self.size)
        elif(sample_last_n):
            idxs = np.arange(batch_size)+self.ptr-batch_size
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=[torch.as_tensor(self.obs_s_buf[i][idxs], dtype=torch.float32, device=self.device) for i in  range(self.N)],
                     obs2=[torch.as_tensor(self.obs2_s_buf[i][idxs], dtype=torch.float32, device=self.device) for i in  range(self.N)],
                     act=torch.as_tensor(self.act_buf[idxs], dtype=torch.float32, device=self.device),
                     rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32, device=self.device),
                     done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32, device=self.device))
        
        return batch
    def save(self, f="data/buffer"):
        with open(f+".npy", 'wb') as f:
            np.save(f, self.act_buf)
            for i in range(self.N):
                np.save(f, self.obs_s_buf[i])
                np.save(f, self.obs2_s_buf[i])
            np.save(f, self.rew_buf)
            np.save(f, self.done_buf)
            np.save(f, self.N)
            np.save(f, self.ptr)
            np.save(f, self.size)
            np.save(f, self.max_size)

    def load(self, path:str, re_compute_reward=False,
                   u_weight=1.0, s_weight =0, omega_weight=0,
                   dot_s_weight=0, dot_u_weight=0, max_reward=1.0, dt=0.005,
                   money_loss_weight=0.0):
        if(path.split(".")[-1]=="npy"):
            with open(path, 'rb') as f:
                self.act_buf = np.load(f)
                self.obs_s_buf = []
                self.obs2_s_buf = []
                for i in range(self.N):
                    self.obs_s_buf.append(np.load(f))
                    self.obs2_s_buf.append(np.load(f))
                self.rew_buf = np.load(f)
                self.done_buf = np.load(f)
                self.N = np.load(f)
                self.ptr = np.load(f)
                self.size = np.load(f)
                self.max_size = np.load(f)
        else:
            with open(path+'.npy', 'rb') as f:
                self.act_buf = np.load(f)
                self.obs_s_buf = []
                self.obs2_s_buf = []
                for i in range(self.N):
                    self.obs_s_buf.append(np.load(f))
                    self.obs2_s_buf.append(np.load(f))
                self.rew_buf = np.load(f)
                self.done_buf = np.load(f)
                self.N = np.load(f)
                self.ptr = np.load(f)
                self.size = np.load(f)
                self.max_size = np.load(f)
        if(re_compute_reward):
            u = self.act_buf
            omega = np.array([self.obs_s_buf[i][:,1] for i in range(self.N)]).T
            s = np.array([self.obs_s_buf[i][:,2] for i in range(self.N)]).T
            s_2 =  np.array([self.obs2_s_buf[i][:,2] for i in range(self.N)]).T
            dot_s =  (s_2-s)/dt
            dot_u =  np.array([self.obs2_s_buf[i][:,7] for i in range(self.N)]).T
            
            u_loss = u_weight*abs(u)
            omega_loss = omega_weight*abs(omega)
            s_loss = s_weight*abs(s)
            dot_s_loss = dot_s_weight*abs(dot_s)
            dot_u_loss = dot_u_weight*abs(dot_u)
            F_of_game(None, N=self.N)
            money_loss = money_loss_weight*F_of_game(u,is_batch=True)
            loss = u_loss**2+omega_loss**2+s_loss**2+dot_s_loss**2+dot_u_loss**2+money_loss

            # loss = u_weight*u*u+s_weight*s*s+omega_weight*omega*omega+\
            #         dot_s_weight*dot_s*dot_s+dot_u_weight*dot_u*dot_u
            self.rew_buf[:] = dt*(max_reward-loss)
            self.else_buf = {"u_loss":dt*u_loss**2,
                             "s_loss":dt*s_loss**2,
                             "omega_loss": dt*omega_loss**2,
                             "dot_s_loss": dt*dot_s_loss**2,
                             "dot_u_loss": dt*dot_u_loss**2,
                             "money_loss": dt*money_loss}
            
            # ok_done = np.logical_and(self.done_buf, np.max(omega, axis=1, keepdims=True)<0.00005)
            # print(ok_done.shape)
            # print(self.rew_buf.shape)
            # print(max_reward)
            # self.rew_buf[ok_done.flatten(), :] += 20*max_reward

if(__name__ == "__main__"):
    a = ReplayBuffer([3,4], [1,1], 10)
    # a.load("data/buffer")
    a.store([np.array([1.0, 2, 3]), np.array([1.0, 2, 3, 4])], 
            np.array([1.2,1.3]),
             np.array([1.0, 1.2]), 
             [np.array([1.0, 2, 3]), np.array([1.0, 2, 3, 4])], 
             np.array([1.0]))
    a.save()
    print(a)
    a.load("data/buffer")

