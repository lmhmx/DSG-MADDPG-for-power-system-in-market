from Env_Closedloop_IEEE_Standard_Bus_Model import Env_Closedloop_IEEE_Standard_Bus_Model
from Policy_Linear_Controller import Policy_Linear_Controller
from Policy_Controller_base import  Policy_Controller_base
from Policy_Piecewise_Controller import Policy_Piecewise_Controller
import torch
from torch.optim import Adam
import torch.nn as nn
from nabla_F_of_game import nabla_F_of_game, F_of_game
from Q_Functions_for_MADDPG import Q_Functions_for_MADDPG
from copy import deepcopy
from ReplayBuffer import ReplayBuffer
import datetime
import numpy as np
from DataRecorder import DataRecorder
import matplotlib.pyplot as plt
from time_str import time_str, time_int, Print_Logger
from set_rand_seed import set_rand_seed
import sys
import os
import argparse
import wandb

class ActorCritic_s(nn.Module):
    def __init__(self, N, obs_num_s, pi_net: Policy_Controller_base, q_nets:Q_Functions_for_MADDPG) -> None:
        super().__init__()
        self.pi_s = pi_net
        self.q_s = q_nets
    def act(self, obs_s):
        with torch.no_grad():
            return self.pi_s(obs_s).cpu().numpy()
        
class MADDDPG:
    def __init__(self, device, gamma=0.99, polyak=0.99, pre_replay_buffer=None, lr=0.001, 
                 omega_weight=200000, u_weight=2.0, s_weight=2.0, s_dot_weight = 1.0,
                  u_dot_weight = 1.0, max_reward=2.0, re_compute_reward = False, 
                  pre_trained_model = None, update_target_every = 1,
                  run_init_param=False, u_overline = None, u_underline = None,
                  N = 3, use_linear_policy = True, money_loss_weight=0.0,
                  replay_buffer_size = 1e6, d=4, save_path = "model"):
        self.device = device
        self.env = Env_Closedloop_IEEE_Standard_Bus_Model(nabla_F=nabla_F_of_game, F_of_game=F_of_game,
                                                          omega_weight=omega_weight, u_weight=u_weight, s_weight=s_weight,
                                                           s_dot_weight=s_dot_weight, u_dot_weight=u_dot_weight, max_reward=max_reward,
                                                           u_overline=u_overline,u_underline=u_underline, N=N,
                                                           money_loss_weight=money_loss_weight)
        self.test_env = Env_Closedloop_IEEE_Standard_Bus_Model(nabla_F=nabla_F_of_game, F_of_game=F_of_game,
                                                               omega_weight=omega_weight, u_weight=u_weight, s_weight=s_weight,
                                                                s_dot_weight=s_dot_weight, u_dot_weight=u_dot_weight, 
                                                                max_reward=max_reward,
                                                                u_overline=u_overline,u_underline=u_underline, N=N,
                                                                money_loss_weight=money_loss_weight)
        if(use_linear_policy):
            self.policy = Policy_Linear_Controller(self.env.N, device=self.device)
        else:
            self.policy = Policy_Piecewise_Controller(self.env.N, d=d, device=self.device)
        self.qf_s = Q_Functions_for_MADDPG(self.env.N, self.env.obs_num_s, self.env.action_num_s, self.env.neighbours, device=self.device)
        self.ac_s = ActorCritic_s(self.env.N, self.env.obs_num_s, self.policy, self.qf_s)
        self.gamma = gamma
        self.polyak = polyak
        self.max_time = 15
        self.max_it_length = int(self.max_time/self.env.dt)

        self.ac_targ_s = deepcopy(self.ac_s)
        for p in self.ac_targ_s.parameters():
            p.requires_grad = False
        
        self.replay_buffer = ReplayBuffer(self.env.obs_num_s, self.env.action_num_s, replay_buffer_size,
                                           device=self.device, load_path=pre_replay_buffer,
                                           re_compute_reward=re_compute_reward,
                                           u_weight=u_weight, omega_weight=omega_weight,
                                            dot_s_weight=s_dot_weight,dot_u_weight=u_dot_weight,
                                            max_reward=max_reward,dt=self.env.dt )
        
        self.pi_optimizer = Adam(self.ac_s.pi_s.parameters(), lr = lr)
        self.q_optimizers = Adam(self.ac_s.q_s.parameters(), lr = lr)
        self.update_target_every = update_target_every
        self.save_path = save_path
        if(pre_trained_model):
            self.load_models(name=pre_trained_model)
        elif(run_init_param):
            self.init_ac_param()
            self.ac_targ_s = deepcopy(self.ac_s)
            for p in self.ac_targ_s.parameters():
                p.requires_grad = False
            self.q_optimizers = Adam(self.ac_s.q_s.parameters(), lr = lr)
        else:
            pass
        self.save_models("model-init-model")

    def init_ac_param(self):
        print("running Q with the parameters of pi of:  {:10.0f}\n K: {}".format(time_int(), self.ac_s.pi_s.K.T))
        q_buf_s = []
        o_buf_s = []
        a_buf_s = []
        # p_range = [-1, -0.75, 0.75, 1]
        p_range = [-1, 1]
        for i in p_range:
            q_buf_1, o_buf_1, a_buf_1 = self.init_ac_param_label(p = self.env.power_model._p.copy()*i)
            q_buf_s.append(q_buf_1)
            o_buf_s.append(o_buf_1)
            a_buf_s.append(a_buf_1)
        # q_buf_2, o_buf_2, a_buf_2 = self.init_ac_param_label(p = -self.env.power_model._p.copy())

        for init_train_epoch in range(2000):
            self.q_optimizers.zero_grad()
            loss_q = 0
            for i in range(len(p_range)):
                q_cal = self.ac_s.q_s.forward(o_buf_s[i], a_buf_s[i]*(1+0.1*(torch.randn_like(a_buf_s[i]))))
                q_cal = torch.hstack(q_cal)
                loss_q = loss_q+((q_buf_s[i]-q_cal)**2).mean()
            # q_cal_2 = self.ac_s.q_s.forward(o_buf_2, a_buf_2*(1+0.1*(torch.randn_like(a_buf_2))))
            # q_cal_2 = torch.hstack(q_cal_2)
            
            loss_q.backward()
            self.q_optimizers.step()
            if(init_train_epoch % 500 ==0):
                print("init_epoch: {:4.0f}, init_loss: {:10.8f} {:10.0f}".format(init_train_epoch, loss_q.item(), time_int()))
                wandb.log({"init_epoch":init_train_epoch, "init_loss":loss_q.item()})
 

    def init_ac_param_label(self, p):
        obs_s = self.env.reset(update_p=True, p=p)
        ep_return = 0
        it_show_progress = max(int(self.max_it_length/20),1)
        init_buffer = ReplayBuffer(self.env.obs_num_s, self.env.action_num_s,
                                   self.max_it_length, self.device, None, False)
        
        for t in range(self.max_it_length):
            a = self.ac_s.act(obs_s)
            obs2_s, r, d, _ = self.env.step(a)
            ep_return += r
            d = 1.0 if t+1 == self.max_it_length else d

            init_buffer.store(obs_s, a, r, obs2_s, d)
            obs_s = obs2_s
            if(d):
                print("done, return: {:7.2f}, ep_len: {:5.0f}, average return: {:10.4f},  {:10.0f}".format(
                            ep_return.mean(), t+1, ep_return.mean()/(t+1), time_int()))
                q_buf = torch.zeros([t+1, self.env.N], device=self.device)
                data = init_buffer.sample_batch(0, sample_all=True)
                o_buf, a_buf, r_buf = data['obs'], data['act'], data['rew']
                q_buf[t] = r_buf[t]
                for q_buf_t in reversed(range(0,t)):
                    q_buf[q_buf_t] = r_buf[q_buf_t]+self.gamma*q_buf[q_buf_t+1]

                # for init_train_epoch in range(10000):
                #     self.q_optimizers.zero_grad()
                #     q_cal = self.ac_s.q_s.forward(o_buf, a_buf*(1+0.1*(torch.randn_like(a_buf))))
                #     q_cal = torch.hstack(q_cal)
                #     loss_q = ((q_buf-q_cal)**2).mean()
                #     loss_q.backward()
                #     self.q_optimizers.step()
                #     if(init_train_epoch % 1000 ==0):
                #         print("init_epoch: {:4.0f}, init_loss: {:7.5f} {:10.0f}".format(init_train_epoch, loss_q.item(), time_int()))
                #         wandb.log({"init_epoch":init_train_epoch, "init_loss":loss_q.item()})
                return q_buf, o_buf, a_buf
            
            if(t%it_show_progress==0):
                print("progress: {:6.2f}% {:10.0f}".format(t/self.max_it_length*100, time_int()))
        

    def compute_loss_q(self, data)->torch.Tensor:
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.ac_s.q_s.forward(o,a)
        with torch.no_grad():
            q_pi_targ = self.ac_targ_s.q_s.forward(o2, (1+0.0*(torch.randn_like(a)))*self.ac_targ_s.pi_s.forward(o2, is_batch=True))
            q_pi_targ = torch.hstack(q_pi_targ)
            backup = r + self.gamma * (1-d)*q_pi_targ
        q = torch.hstack(q)
        loss_q = ((q-backup)**2).mean()
        return loss_q
    def compute_loss_pi(self, data):
        o = data["obs"]
        q_pi = self.ac_s.q_s.forward(o, self.ac_s.pi_s.forward(o, is_batch=True))
        q_pi = torch.hstack(q_pi)
        return -q_pi.mean()
    def update_q_network(self, data):
        self.q_optimizers.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizers.step()
        return loss_q.item()
    
    def update_pi_network(self, data):
        for p in self.ac_s.q_s.parameters():
            p.requires_grad = False
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        for p in self.ac_s.q_s.parameters():
            p.requires_grad = True
        return loss_pi.item()
    
    def update(self, batch_size):
        loss_q_s = []
        # q_it_max = 5000
        # q_it_batch_delta = 200
        q_it_max = 200
        for q_it in range(q_it_max):
            data = self.replay_buffer.sample_batch(batch_size)
            loss_q = self.update_q_network(data)
            loss_q_s.append(loss_q)

        data = self.replay_buffer.sample_batch(batch_size)
        loss_pi = self.update_pi_network(data)
        
        with torch.no_grad():
            for p, p_targ in zip(self.ac_s.parameters(), self.ac_targ_s.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1-self.polyak)*p.data)
        return loss_q, loss_pi
    
    def run(self, epoch, batch_size = 128, train_num_every = 100, noise_level = 0.1):
        for i in range(2):
            self.run_data_to_replay_buffer(self.ac_s, noise_level=noise_level)
        for i in range(epoch):
            self.run_data_to_replay_buffer(self.ac_s, noise_level=noise_level)
            self.train_from_replay_buffer(batch_size=batch_size, train_num_every=train_num_every)
            if(i%10==0):
                print("saving model {:3.0f}".format(time_int()), end=" ")
                self.save_models("model-epoch{:0>3d}-".format(i))
                print("saving buffer {:3.0f}".format(time_int()), end=" ")
                self.replay_buffer.save()
                print("saved")
                print("epoch: {:10.0f}/{:10.0f} {:10.0f}".format(i, epoch, time_int()))
                ep_return_test, ep_length_test = self.test_agent()
                wandb.log({"test_return": ep_return_test.mean(),
                           "test_average_len":ep_length_test,
                           "test_return":ep_return_test.mean(),
                           "test_r_0":ep_return_test[0,0],
                           "test_r_1":ep_return_test[1,0],
                           "test_r_2":ep_return_test[2,0]})
            wandb.log({"run_epoch":i})
    
    def run_train_only(self, epoch, batch_size = 128, train_num_every = 100):
        for i in range(epoch):
            self.train_from_replay_buffer(batch_size=batch_size, train_num_every=train_num_every)
            self.save_models("model-epoch{:0>3d}-".format(i))
            print("epoch: {:10.0f}/{:10.0f} {:10.0f}".format(i, epoch, time_int()))
            wandb.log({"run_epoch":i})

    def run_data_only(self, noise_level=0.0):
        # For Linear_policy only
        K_values = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]

        with torch.no_grad():
            for K_value in K_values:
                self.ac_s.pi_s.K_[:] = K_value
                self.run_data_to_replay_buffer(self.ac_s, noise_level=noise_level)
                print("K: {:3.2f} {:10.0f}".format(K_value, time_int()))

    def train_from_replay_buffer(self, batch_size=128, train_num_every = 100):    
        tmp_loss_q = []
        tmp_loss_pi = []
        print("Starting training {:10.0f}".format(time_int()))
        for _ in range(train_num_every):
            loss_q_s, loss_pi_s = self.update(batch_size)
            tmp_loss_q.append(loss_q_s)
            tmp_loss_pi.append(loss_pi_s)
        # return tmp_loss_q, tmp_loss_pi, 
        print("loss_pi: {:7.5f}, loss_q: {:7.5f},  {:10.0f}".format(
                    np.mean(tmp_loss_pi) ,np.mean(tmp_loss_q), time_int()))
        print("loss_q_s: {}".format(np.array(tmp_loss_q, dtype=np.float32).flatten()))
        print("loss_pi_s: {}".format(np.array(tmp_loss_pi, dtype=np.float32).flatten()))
        print("K1: {}".format(self.ac_s.pi_s.K.T))
        print("K_target: {}".format(self.ac_targ_s.pi_s.K.T))
        wandb.log({"train_loss_pi":np.mean(tmp_loss_pi),"train_loss_q":np.mean(tmp_loss_q)})
        wandb.log({"K_max":torch.max(self.ac_s.pi_s.K.T),
                   "K_min":torch.min(self.ac_s.pi_s.K.T),
                   "K_target_max":torch.max(self.ac_targ_s.pi_s.K.T),
                   "K_target_min":torch.min(self.ac_targ_s.pi_s.K.T),
                   "K_0":self.ac_s.pi_s.K[0],
                   "K_1":self.ac_s.pi_s.K[1],
                   "K_2":self.ac_s.pi_s.K[2],
                   "K_t_0":self.ac_targ_s.pi_s.K[0],
                   "K_t_1":self.ac_targ_s.pi_s.K[1],
                   "K_t_2":self.ac_targ_s.pi_s.K[2],})
    def run_data_to_replay_buffer(self, ac_s: ActorCritic_s, noise_level = 0.1)->None:
        print("running a data by the parameters of:  {:10.0f}\n K: {}".format(time_int(), ac_s.pi_s.K.T))
        obs_s = self.env.reset(update_p=True)
        ep_return = 0
        it_show_progress = max(int(self.max_it_length/20),1)
        for t in range(self.max_it_length):
            a = ac_s.act(obs_s)
            a_noise = noise_level*np.random.randn(self.env.N,1)
            a = a + a_noise
            obs2_s, r, d, _ = self.env.step(a, a_noise)
            ep_return += r
            d = 1.0 if t+1 == self.max_it_length else d

            self.replay_buffer.store(obs_s, a, r, obs2_s, d)
            obs_s = obs2_s
            if(d):
                print("done, return: {:7.2f}, ep_len: {:5.0f}, average return: {:10.4f},  {:10.0f}".format(
                            ep_return.mean(), t+1, ep_return.mean()/(t+1), time_int()))
                print("return 0: {:10.7} 0: {:10.7} 0: {:10.7} {:10.0f}".format(
                            float(ep_return[0,0]), ep_return[1,0], ep_return[2,0], time_int()))
                
                wandb.log({"return": ep_return.mean(),
                           "average_len":t+1,
                           "average return":ep_return.mean()/(t+1),
                           "r_0":ep_return[0,0],
                           "r_1":ep_return[1,0],
                           "r_2":ep_return[2,0]})
                return
            if(t%it_show_progress==0):
                print("progress: {:6.2f}% {:10.0f}".format(t/self.max_it_length*100, time_int()))

    def test_agent(self):
        o, d, ep_len = self.test_env.reset(), False, 0
        ep_ret = np.zeros([self.env.N, 1])
        while(not (d or (ep_len == self.max_it_length))):
            o, r, d, _ = self.test_env.step(self.ac_s.act(o))
            ep_len += 1
            ep_ret += r
        return ep_ret, ep_len
    def evaluate_model(self, name):
        self.load_models(name)
        o = self.test_env.reset()
        data_recorder = DataRecorder()
        for i in range(self.max_it_length):
            u = self.ac_s.act(o)
            o, _, _, _ = self.test_env.step(u)
            theta = np.array([[o[i][0] for i in range(self.test_env.N) ]])
            omega = np.array([[o[i][1] for i in range(self.test_env.N) ]])
            s = np.array([[o[i][2] for i in range(self.test_env.N) ]])
            nu_minus = np.array([[o[i][3] for i in range(self.test_env.N) ]])
            nu_plus = np.array([[o[i][4] for i in range(self.test_env.N) ]])
            lambda_ = np.array([[o[i][5] for i in range(self.test_env.N) ]])
            mu_ = np.array([[o[i][6] for i in range(self.test_env.N) ]])
            data_recorder.add(theta, "theta")
            data_recorder.add(omega, "omega")
            data_recorder.add(s, "s")
            data_recorder.add(nu_minus, "nu_minus")
            data_recorder.add(nu_plus, "nu_plus")
            data_recorder.add(lambda_, "lambda")
            data_recorder.add(mu_, "mu")
            data_recorder.add(u, "u")
        omega_s = data_recorder.get("omega")
        theta_s = data_recorder.get("theta")
        s_s = data_recorder.get("s")
        nu_minus_s = data_recorder.get("nu_minus")
        nu_plus_s = data_recorder.get("nu_plus")
        lambda_s = data_recorder.get("lambda")
        mu_s = data_recorder.get("mu")
        u_s = data_recorder.get("u")
        
        print(self.ac_s.pi_s.state_dict())
        plt_m = 3
        plt_n = 3
        plt.figure(figsize=[plt_n*3, plt_m*3])
        plt.subplot(plt_m,plt_n,1)
        plt.plot(omega_s)
        plt.ylabel("omega")
        plt.subplot(plt_m,plt_n,2)
        plt.plot(theta_s)
        plt.ylabel("theta")
        plt.subplot(plt_m,plt_n,3)
        plt.plot(s_s)
        plt.ylabel("s")
        plt.subplot(plt_m,plt_n,4)
        plt.plot(nu_minus_s)
        plt.ylabel("nu_m")
        plt.subplot(plt_m,plt_n,5)
        plt.plot(nu_plus_s)
        plt.ylabel("nu_p")
        plt.subplot(plt_m,plt_n,6)
        plt.plot(lambda_s)
        plt.ylabel("lambda")
        plt.subplot(plt_m,plt_n,7)
        plt.plot(mu_s)
        plt.ylabel("mu")
        plt.subplot(plt_m,plt_n,8)
        plt.plot(u_s)
        plt.ylabel("u")
        plt.subplot(plt_m,plt_n,9)
        plt.ylabel("omega")
        plt.plot(omega_s[:, 52:68])
        plt.savefig("./fig/"+time_str()+".png")
        
    def save_models(self, name="model", add_extra_infos = True):
        if(add_extra_infos):
            if not os.path.exists("./model/"+self.save_path):
                os.mkdir("./model/"+self.save_path)
            time_str_current = time_str()
            torch.save(self.ac_s.pi_s.state_dict(), "./model/"+self.save_path+"/"+time_str_current+name+"pi_s"+".pth")
            torch.save(self.ac_s.q_s.state_dict(), "./model/"+self.save_path+"/"+time_str_current+name+"q_s"+".pth")
            torch.save(self.ac_targ_s.pi_s.state_dict(), "./model/"+self.save_path+"/"+time_str_current+name+"targ_pi_s"+".pth")
            torch.save(self.ac_targ_s.q_s.state_dict(), "./model/"+self.save_path+"/"+time_str_current+name+"targ_q_s"+".pth")
        else:
            torch.save(self.ac_s.pi_s.state_dict(), "./model/"+name+"pi_s"+".pth")
            torch.save(self.ac_s.q_s.state_dict(), "./model/"+name+"q_s"+".pth")
            torch.save(self.ac_targ_s.pi_s.state_dict(), "./model/"+name+"targ_pi_s"+".pth")
            torch.save(self.ac_targ_s.q_s.state_dict(), "./model/"+name+"targ_q_s"+".pth")

    def load_models(self, name):
        self.ac_s.pi_s.load_state_dict(torch.load(name+"pi_s"+".pth"))
        self.ac_s.q_s.load_state_dict(torch.load(name+"q_s"+".pth"))
        self.ac_targ_s.pi_s.load_state_dict(torch.load(name+"targ_pi_s"+".pth"))
        self.ac_targ_s.q_s.load_state_dict(torch.load(name+"targ_q_s"+".pth"))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cmd_args", type=str, default="n")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--train_num_every", type=int, default=200)
    parser.add_argument("--pre_replay_buffer", type=str, 
                        help="the type will be transformed to None type when given as None or none", 
                        default="None")
    parser.add_argument("--run_train_only", type=str, 
                        help="this will be changed to a bool in the code", default="n")
    parser.add_argument("--max_reward", type=float,help="max_reward", default=1.0)
    parser.add_argument("--u_weight", type=float, default=1.0)
    parser.add_argument("--omega_weight", type=float, default=10*10000)
    parser.add_argument("--s_weight", type=float, default=0.0)
    parser.add_argument("--u_dot_weight", type=float, default=15.0)
    parser.add_argument("--s_dot_weight", type=float, default=0.0)
    parser.add_argument("--money_loss_weight", type=float, default=0.0)
    parser.add_argument("--re_compute_reward",help="this will be changed to a bool in the code",
                         type=str, default="n")
    parser.add_argument("--run_data_only", type=str, default="n")
    parser.add_argument("--pre_trained_model", type=str, default="none")
    parser.add_argument("--update_target_every", type=int,help="not used any more", default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_init_param", type=str, default="n")
    parser.add_argument("--u_limit", type=str, default="n")
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--use_linear_policy", type=str, default="n")
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6))
    parser.add_argument("--d", type=int, default=int(4))
    # code_args = "--device cuda --gamma 0.99 --polyak 0.9999"+\
    #             " --lr 0.01 --batch_size 2048 --epoch 50" + \
    #             " --train_num_every 50" +\
    #             " --pre_replay_buffer useful/buffer20231006-19-35-11.npy"+\
    #             " --s_weight 0.0 --u_weight 0.0 --omega_weight 500"+\
    #             " --s_dot_weight 5 --u_dot_weight 5 --max_reward 100.0"+\
    #             " --re_compute_reward y --run_train_only y"+\
    #             " --run_data_only n --update_target_every 20"+ \
    #             " --seed 42" +\
    #             " --pre_trained_model useful/model-epoch049-20231008-20-02-08"
    # code_args = "--device cuda --gamma 0.999 --polyak 0.8"+\
    #             " --lr 0.005 --batch_size 8192 --epoch 1500" + \
    #             " --train_num_every 1" +\
    #             " --pre_replay_buffer useful/buffer20231127-17-22-56.npy"+\
    #             " --s_weight 0.0 --u_weight 0.0 --omega_weight 1000"+\
    #             " --s_dot_weight 0.0 --u_dot_weight 0.0 --max_reward 0.0"+\
    #             " --re_compute_reward n --run_train_only n"+\
    #             " --run_data_only n --update_target_every 2"+ \
    #             " --seed 42 --run_init_param y" +\
    #             " --pre_trained_model useful/20231127-17-22-54model-epoch999-" +\
    #             " --u_limit n --N 3"
    code_args = "--use_cmd_args n"+\
                " --device cuda --gamma 0.999 --polyak 0.95"+\
                " --lr 0.005 --batch_size 8192 --epoch 500" + \
                " --train_num_every 1" +\
                " --pre_replay_buffer None"+\
                " --s_weight 0.0 --u_weight 0.0 --omega_weight 200"+\
                " --s_dot_weight 0.0 --u_dot_weight 0.5 --max_reward 0.0"+\
                " --money_loss_weight 0.0" +\
                " --re_compute_reward n --run_train_only n"+\
                " --run_data_only n --update_target_every 2"+ \
                " --seed 15 --run_init_param y" +\
                " --pre_trained_model None" +\
                " --u_limit n --N 68 --noise_level 0.03" +\
                " --use_linear_policy n" +\
                " --replay_buffer_size 90000" +\
                " --d 4"
    
    args = parser.parse_args()
    if(args.use_cmd_args == "n" or args.use_cmd_args == "N"):
        args = parser.parse_args(code_args.split(" "))

    args.pre_replay_buffer = None if(args.pre_replay_buffer == "None" or args.pre_replay_buffer == "none") else args.pre_replay_buffer
    args.run_train_only = False if(args.run_train_only == "n") else True
    args.run_data_only = False if(args.run_data_only == "n") else True
    args.run_init_param = False if(args.run_init_param == "n") else True
    args.use_linear_policy = False if(args.use_linear_policy == "n") else True
    args.re_compute_reward = False if(args.re_compute_reward =="n") else True    
    args.pre_trained_model = None if(args.pre_trained_model == "None" or args.pre_trained_model == "none") else args.pre_trained_model
    args.u_overline = 10000*np.ones([args.N, 1], dtype = np.float32)
    args.u_overline[[0,1,2], 0] = args.u_overline[[0,1,2], 0] if(args.u_limit == "n" or args.u_limit == "N") else [0.5, 0.4, 0.3]
    args.u_underline = -args.u_overline
    args.run_name = "68-bus with different initial state"
    return args

if(__name__ == "__main__"):
    sys.stdout = Print_Logger()
    args = get_args()
    set_rand_seed(args.seed)
    print("args: {}".format(args))
    wandb_id = time_str()
    wandb.init(project="GNE_In_Power_System",
               config=vars(args),
               id=wandb_id)
    ma = MADDDPG(device=args.device, 
                 gamma=args.gamma,
                 polyak=args.polyak,
                 pre_replay_buffer=args.pre_replay_buffer,
                 lr=args.lr,
                 omega_weight=args.omega_weight,
                 u_weight=args.u_weight, 
                 s_weight=args.s_weight, 
                 u_dot_weight=args.u_dot_weight,
                 s_dot_weight=args.s_dot_weight,
                 max_reward=args.max_reward,
                 re_compute_reward=args.re_compute_reward,
                 pre_trained_model=args.pre_trained_model,
                 update_target_every = args.update_target_every,
                 run_init_param=args.run_init_param,
                 u_overline=args.u_overline,u_underline=args.u_underline,
                 N=args.N,
                 use_linear_policy=args.use_linear_policy,
                 money_loss_weight=args.money_loss_weight,
                 replay_buffer_size=args.replay_buffer_size,
                 d=args.d, save_path=wandb_id)
    if(args.run_train_only):
        ma.run_train_only(epoch=args.epoch, batch_size=args.batch_size, train_num_every=args.train_num_every)
    elif(args.run_data_only):
        ma.run_data_only(noise_level=args.noise_level)
        ma.replay_buffer.save("data/buffer"+time_str())
        ma.replay_buffer.save("data/buffer")
    else:
        ma.run(epoch=args.epoch, batch_size=args.batch_size,
                train_num_every=args.train_num_every, noise_level = args.noise_level)
        ma.save_models(add_extra_infos=False)
        # restore 2 bakes. The first one is to record. The second one is to 
        ma.replay_buffer.save("data/buffer"+time_str())
        ma.replay_buffer.save("data/buffer")
    wandb.finish()
    # ma.load_models("model-epoch920230916-00-53-22")
    # ma.test_agent()

    # ma.evaluate_model("./model/20230921-14-02-16/model20230921-17-03-34")
    # print("finish")
