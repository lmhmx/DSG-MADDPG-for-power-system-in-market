from Env_Closedloop_IEEE_Standard_Bus_Model import Env_Closedloop_IEEE_Standard_Bus_Model
from Policy_Piecewise_Controller import Policy_Piecewise_Controller
from sko.GA import GA
from sko.PSO import PSO
from nabla_F_of_game import nabla_F_of_game, F_of_game
import torch
import numpy as np
from time_str import time_str, time_int, Print_Logger
from DataRecorder import DataRecorder
import sys
import wandb
import copy
from set_rand_seed import set_rand_seed

class Policy_Evaluate:
    def __init__(self, omega_weight, u_weight, s_weight,
                        s_dot_weight, u_dot_weight, max_reward,
                        u_overline,u_underline, N,
                        money_loss_weight, d, use_wandb):
        self.env = Env_Closedloop_IEEE_Standard_Bus_Model(nabla_F=nabla_F_of_game, F_of_game=F_of_game,
                                                          omega_weight=omega_weight, u_weight=u_weight, s_weight=s_weight,
                                                           s_dot_weight=s_dot_weight, u_dot_weight=u_dot_weight, max_reward=max_reward,
                                                           u_overline=u_overline,u_underline=u_underline, N=N,
                                                           money_loss_weight=money_loss_weight)
        self.test_env = Env_Closedloop_IEEE_Standard_Bus_Model(nabla_F=nabla_F_of_game, F_of_game=F_of_game,
                                                          omega_weight=omega_weight, u_weight=u_weight, s_weight=s_weight,
                                                           s_dot_weight=s_dot_weight, u_dot_weight=u_dot_weight, max_reward=max_reward,
                                                           u_overline=u_overline,u_underline=u_underline, N=N,
                                                           money_loss_weight=money_loss_weight)
        self.policy = Policy_Piecewise_Controller(N = N, d = d, device='cpu')

        self.max_time = 15
        self.max_it_length = int(self.max_time/self.env.dt)
        self.evaluate_num = 1
        self.use_wandb = use_wandb
        self.recorder = DataRecorder()
        self.recorder.add(np.array(N), "N")
        self.recorder.add(np.array(d), "d")

    def policy_evaluate(self, env:Env_Closedloop_IEEE_Standard_Bus_Model, 
                        policy:Policy_Piecewise_Controller, update_p = True):
        info_str = "Train" if (update_p == True) else "Test"
        print("{}ing agents {:7.0f} {:10.0f} ".format(info_str, self.evaluate_num, time_int()),end = " ")
        if(not update_p):
            print(policy.b_minus)
            print(policy.b_plus)
            print(policy.k_minus)
            print(policy.k_plus)
        self.evaluate_num += 1
        o, d, ep_len = env.reset(update_p=update_p), False, 0
        ep_ret = np.zeros([self.env.N, 1])
        while(not (d or (ep_len == self.max_it_length))):
            o, r, d, _ = env.step(policy.forward(o).cpu().numpy())
            ep_len += 1
            ep_ret += r
        print("{} end {:10.0f}".format(info_str, time_int()))
        return -ep_ret, ep_len
    def test_obj_func(self, par):
        par = np.power(10, np.array(par))
        N = self.policy.N
        d = self.policy.d
        index_num = [0,N*d,N*d,N*(d-1),N*(d-1)]
        index = [int(np.sum(index_num[0:i+1])) for i in range(len(index_num))]
        with torch.no_grad():
            self.policy.kappa_minus[:,:] = torch.tensor(np.reshape(par[index[0]:index[1]], [N, d]))
            self.policy.kappa_plus[:,:]  = torch.tensor(np.reshape(par[index[1]:index[2]], [N, d]))
            self.policy.beta_minus[:,:]  = torch.tensor(np.reshape(par[index[2]:index[3]], [N, d-1]))
            self.policy.beta_plus[:,:]   = torch.tensor(np.reshape(par[index[3]:index[4]], [N, d-1]))
            
            scores, _ = self.policy_evaluate(self.test_env, self.policy, update_p=False)

        print("{} {:10.7f}".format(scores.flatten(), scores.mean()))
        return np.mean(scores)
    
    def obj_func(self, par):
        par = np.power(10, np.array(par))
        N = self.policy.N
        d = self.policy.d
        index_num = [0,N*d,N*d,N*(d-1),N*(d-1)]
        index = [int(np.sum(index_num[0:i+1])) for i in range(len(index_num))]
        with torch.no_grad():
            self.policy.kappa_minus[:,:] = torch.tensor(np.reshape(par[index[0]:index[1]], [N, d]))
            self.policy.kappa_plus[:,:]  = torch.tensor(np.reshape(par[index[1]:index[2]], [N, d]))
            self.policy.beta_minus[:,:]  = torch.tensor(np.reshape(par[index[2]:index[3]], [N, d-1]))
            self.policy.beta_plus[:,:]   = torch.tensor(np.reshape(par[index[3]:index[4]], [N, d-1]))
            
            scores, _ = self.policy_evaluate(self.env, self.policy, update_p=True)

        print("{} {:10.7f}".format(scores.flatten(), scores.mean()))
        if(self.use_wandb):
            wandb.log({"epoch":self.evaluate_num,
                       "score":-np.mean(scores),
                       "score_0":-scores[0],
                       "score_1":-scores[1],
                       "score_2":-scores[2],
                       })
        self.recorder.add(np.array(scores), "scores")
        self.recorder.add(np.array(par), "param")
        return np.mean(scores)

def run_search(policy_evaluate:Policy_Evaluate, search_method, lb, ub):
    if(search_method == "GA"):
        size_pop = 10
        run_iter = 41
        g = GA(policy_evaluate.obj_func, n_dim=len(lb), size_pop=size_pop, max_iter=100000,
                prob_mut=0.005, lb=lb, ub=ub, precision=1e-5)
        g.run(run_iter)
        policy_evaluate.recorder.add(np.array(size_pop), "size_pop")
        policy_evaluate.recorder.add(np.array(run_iter), "run_iter")
        policy_evaluate.recorder.add(np.array(g.generation_best_X), "generation_best_X")
        policy_evaluate.recorder.add(np.array(g.generation_best_Y), "generation_best_Y")
        test_scores = []
        for par in g.generation_best_X:
            test_scores.append(policy_evaluate.test_obj_func(par))
        policy_evaluate.recorder.add(np.array(test_scores), "test_scores")

    elif(search_method == "PSO"):
        size_pop = 10
        run_iter = 41
        p = PSO(func=policy_evaluate.obj_func, n_dim=len(lb), pop=size_pop, max_iter=100000, 
                    lb=lb, ub=ub, w=0.3, c1=0.5, c2=0.5)
        p.record_mode = True
        p.run(run_iter)
        policy_evaluate.recorder.add(np.array(size_pop), "size_pop")
        policy_evaluate.recorder.add(np.array(run_iter), "run_iter")
        policy_evaluate.recorder.add(np.array(p.record_value["X"]), "X")
        policy_evaluate.recorder.add(np.array(p.record_value["Y"]), "Y")
        test_scores = []
        for t in range(len(p.record_value["X"])):
            best_y_index = np.argmin(p.record_value["Y"][t])
            test_scores.append(policy_evaluate.test_obj_func(p.record_value["X"][t][best_y_index,:]))
        policy_evaluate.recorder.add(np.array(test_scores), "test_scores")
    return policy_evaluate

def main_run_PSO(args, use_wandb):
    set_rand_seed(args.seed)
    policy_evaluate_PSO = Policy_Evaluate(
        omega_weight=args.omega_weight, u_weight=args.u_weight, s_weight=args.s_weight,
        s_dot_weight=args.s_dot_weight, u_dot_weight=args.u_dot_weight, max_reward=args.max_reward,
        u_overline=args.u_overline,u_underline=args.u_underline, N=args.N,
        money_loss_weight=args.money_loss_weight, d = args.d, use_wandb=use_wandb
    )
    kappa_range = [-0.4, 0.7]
    beta_range = [-1, 0]
    N = args.N
    d = args.d

    policy_evaluate_PSO = run_search(policy_evaluate_PSO, "PSO", 
               lb = [kappa_range[0]]*N*d*2+[beta_range[0]]*N*(d-1)*2,
               ub = [kappa_range[1]]*N*d*2+[beta_range[1]]*N*(d-1)*2)
    print("PSO search over")
    return policy_evaluate_PSO.recorder

def main_run_GA(args, use_wandb):
    set_rand_seed(args.seed)
    policy_evaluate_GA = Policy_Evaluate(
        omega_weight=args.omega_weight, u_weight=args.u_weight, s_weight=args.s_weight,
        s_dot_weight=args.s_dot_weight, u_dot_weight=args.u_dot_weight, max_reward=args.max_reward,
        u_overline=args.u_overline,u_underline=args.u_underline, N=args.N,
        money_loss_weight=args.money_loss_weight, d = args.d, use_wandb=use_wandb
    )

    kappa_range = [-0.4, 0.7]
    beta_range = [-1, 0]
    N = 68
    d = 4

    policy_evaluate_GA = run_search(policy_evaluate_GA, "GA", 
               lb = [kappa_range[0]]*N*d*2+[beta_range[0]]*N*(d-1)*2,
               ub = [kappa_range[1]]*N*d*2+[beta_range[1]]*N*(d-1)*2)
    print("GA search over")
    return policy_evaluate_GA.recorder

def main_run_PSO_GA():
    sys.stdout = Print_Logger()
    from run_MADDDPG import get_args
    args = get_args()
    use_wandb = True
    if(use_wandb):
        wandb.init(project="DSGMADDPG-power-system-PSO-GA",
                config=vars(args),
                id=time_str())
    ga_data = main_run_GA(args, use_wandb)
    ga_data.save("./record/ga_{}.pkl".format(time_str()))
    pso_data = main_run_PSO(args, use_wandb)
    pso_data.save("./record/pso_{}.pkl".format(time_str()))
    if(use_wandb):
        wandb.finish()
    return pso_data, ga_data

if(__name__ == "__main__"):
    p_r, g_r=main_run_PSO_GA()
    print(p_r.get("test_scores"))
    print(g_r.get("test_scores"))
    p_r.save("./record/pso_{}.pkl".format(time_str()))
    g_r.save("./record/ga_{}.pkl".format(time_str()))
