from IEEE_Standard_Bus_Model import IEEE_Standard_Bus_Model
from DataRecorder import DataRecorder
from PowerSystemController import IntegralController, PiecewiseController
import matplotlib.pyplot as plt
from init_work_space import init_work_space
from nabla_F_of_game import nabla_F_of_game
from time_str import time_str, time_int, search_file_from_path
from main_run_power_system_controller import main_run_power_system_controller
import numpy as np
from run_PSO_GA import main_run_PSO_GA
from run_under_d import run_under_d
import os
import wandb
from Q_Functions_for_MADDPG import Q_Functions_for_MADDPG
from run_MADDDPG import ActorCritic_s
from Env_Closedloop_IEEE_Standard_Bus_Model import Env_Closedloop_IEEE_Standard_Bus_Model
from nabla_F_of_game import nabla_F_of_game, F_of_game
from Policy_Piecewise_Controller import Policy_Piecewise_Controller
import torch
from Policy_Linear_Controller import Policy_Linear_Controller


def plot_controller_result(
        integral_recorder:DataRecorder,
        GNE_recorder_no_constraint:DataRecorder,
        GNE_recorder_constraint:DataRecorder,
        GNE_recorder_constraint_with_hard:DataRecorder,
        dt = 0.005
        ):
    plt_m = 2
    plt_n = 2
    plt.figure(figsize=[plt_n*6, plt_m*4])
    plt.subplot(plt_m, plt_n, 1)
    plt.plot(np.arange(len(integral_recorder.get("omega")))*dt, 60+60*integral_recorder.get("omega")[:, [0,1,2, 52, 53, 54]], "--")
    plt.plot(np.arange(len(GNE_recorder_no_constraint.get("omega")))*dt, 60+60*GNE_recorder_no_constraint.get("omega")[:, [0,1,2, 52, 53, 54]])
    plt.legend(["I 1","I 2","I 3","I 53","I 54","I 55","GNE 1","GNE 2","GNE 3","GNE 53","GNE 54","GNE 55"],ncol=2)
    plt.xlim(0, dt*len(integral_recorder.get("omega")))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.subplot(plt_m, plt_n, 2)
    plt.plot(np.arange(len(integral_recorder.get("u")))*dt, integral_recorder.get("u")[:, [0,1,2, 52, 53, 54]],"--")
    plt.plot(np.arange(len(GNE_recorder_no_constraint.get("u")))*dt, GNE_recorder_no_constraint.get("u")[:, [0,1,2, 52, 53, 54]])
    plt.legend(["I 1","I 2","I 3","I 53","I 54","I 55","GNE 1","GNE 2","GNE 3","GNE 53","GNE 54","GNE 55"],ncol=2)
    plt.xlim(0, dt*len(integral_recorder.get("omega")))
    plt.xlabel("Time (s)")
    plt.ylabel("Controllable Power u (p.u.)")
    plt.subplot(plt_m, plt_n, 3)
    plt.plot(np.arange(len(GNE_recorder_no_constraint.get("u")))*dt, GNE_recorder_no_constraint.get("u")[:, [0,1,2]])
    plt.plot(np.arange(len(GNE_recorder_constraint.get("u")))*dt, GNE_recorder_constraint.get("u")[:, [0,1,2]],"--")
    plt.plot(np.arange(len(GNE_recorder_constraint.get("u")))*dt, np.array([0.5, 0.4, 0.3])+0*GNE_recorder_constraint.get("u")[:, [0,1,2]],"--",color="black",linewidth=1)
    plt.legend(["GNE 1","GNE 2","GNE 3","Cconstraint 1","Cconstraint 2","Cconstraint 3", "Constraint"],ncol=2)
    plt.xlim(0, dt*len(integral_recorder.get("omega")))
    plt.xlabel("Time (s)")
    plt.ylabel("Controllable Power u (p.u.)")
    plt.subplot(plt_m, plt_n, 4)
    plt.plot(np.arange(len(GNE_recorder_constraint_with_hard.get("u")))*dt, GNE_recorder_constraint_with_hard.get("u")[:, [0,1,2]])
    plt.plot(np.arange(len(GNE_recorder_constraint.get("u")))*dt, GNE_recorder_constraint.get("u")[:, [0,1,2]],"--")
    plt.plot(np.arange(len(GNE_recorder_constraint.get("u")))*dt, np.array([0.5, 0.4, 0.3])+0*GNE_recorder_constraint.get("u")[:, [0,1,2]],"--",color="black",linewidth=1)
    plt.plot(np.arange(len(GNE_recorder_constraint.get("u")))*dt, np.array([0.55, 0.45, 0.35])+0*GNE_recorder_constraint.get("u")[:, [0,1,2]],"-",color="black",linewidth=1)
    plt.legend(["Saturated 1","Saturated 2","Saturated 3","Linear 1","Linear 2","Linear 3", "Constraint", "Hard Constraint"],ncol=2)
    plt.xlim(0, dt*len(integral_recorder.get("omega")))
    plt.xlabel("Time (s)")
    plt.ylabel("Controllable Power u (p.u.)")
    plt.savefig("./fig/"+time_str()+"Result.png")
    print("*"*70+"\nDone! 'plot_controller_result' has store the figures\n"+"*"*70)

def plot_pso_ga_result(pso_recorder:DataRecorder,
                        ga_recorder:DataRecorder):
    N = int(pso_recorder.get("N")[0,0])
    d = int(pso_recorder.get("d")[0,0])
    pso_scores = pso_recorder.get("scores")
    pso_size_pop = int(pso_recorder.get("size_pop")[0,0])
    pso_run_iter = int(pso_recorder.get("run_iter")[0,0])
    
    ga_scores = ga_recorder.get("scores")
    ga_size_pop = int(ga_recorder.get("size_pop")[0,0])
    ga_run_iter = int(ga_recorder.get("run_iter")[0,0])
    
    # pso_average_score = -np.mean(pso_scores, axis = 1, keepdims=False)
    # pso_score_every_iter = np.reshape(pso_average_score, [pso_run_iter+1, pso_size_pop])
    # pso_score_every_iter_max = np.max(pso_score_every_iter, axis=1, keepdims=False)
    # pso_score_index = pso_size_pop*np.array(range(0, pso_run_iter+1))

    # ga_average_score = -np.mean(ga_scores, axis = 1, keepdims=False)
    # ga_score_every_iter = np.reshape(ga_average_score[0:ga_run_iter*ga_size_pop], [ga_run_iter, ga_size_pop])
    # ga_score_every_iter_max = np.max(ga_score_every_iter, axis=1, keepdims=False)
    # ga_score_index = ga_size_pop*np.array(range(0, ga_run_iter))
    pso_score_every_iter_max = pso_recorder.get("test_scores")
    ga_score_every_iter_max = ga_recorder.get("test_scores")
    pso_score_index = pso_size_pop*np.array(range(pso_run_iter))
    ga_score_index = ga_size_pop*np.array(range(ga_run_iter))
    plt.figure()
    plt.plot(pso_score_index, pso_score_every_iter_max.flatten())
    plt.plot(ga_score_index, ga_score_every_iter_max.flatten())
    plt.xlabel("Iterations")
    plt.ylabel("Return")
    plt.legend(["PSO", "GA"])
    plt.savefig("./fig/"+time_str()+"pso_ga.png")
    print("*"*70+"\nDone! 'plot_pso_ga_result' has store the figures\n"+"*"*70)

def main_plot_controller_result_without_RL(recorder_version = None, dt=0.005, save_mat=False):
    if(recorder_version is None):
        (integral_recorder,
            GNE_recorder_no_constraint,
            GNE_recorder_constraint,
            GNE_recorder_constraint_with_hard) = main_run_power_system_controller()
        # This code should be commented when not needed
        name_str = "v_no_rl_001"
        integral_recorder.save("./record/{}_integral_Controller_{}.pkl".format(name_str, time_str()))
        GNE_recorder_no_constraint.save("./record/{}_linear_Controller_{}.pkl".format(name_str, time_str()))
        GNE_recorder_constraint.save("./record/{}_linear_Constrained_Controller_{}.pkl".format(name_str, time_str()))
        GNE_recorder_constraint_with_hard.save("./record/{}_saturated_linear_Constrained_Controller_{}.pkl".format(name_str, time_str()))
    else:
        # This path should be changed to the path as needed
        path_root = "./useful/"

        recorder_files = [search_file_from_path(path_root, recorder_version+"_integral_Controller.*pkl"),
                          search_file_from_path(path_root, recorder_version+"_linear_Controller.*pkl"),
                          search_file_from_path(path_root, recorder_version+"_linear_Constrained_Controller.*pkl"),
                          search_file_from_path(path_root, recorder_version+"_saturated_linear_Constrained_Controller.*pkl")]

        integral_recorder = DataRecorder(load_file=recorder_files[0])
        GNE_recorder_no_constraint = DataRecorder(load_file=recorder_files[1])
        GNE_recorder_constraint = DataRecorder(load_file=recorder_files[2])
        GNE_recorder_constraint_with_hard = DataRecorder(load_file=recorder_files[3])
        
    # Save as mat. The mat files can be used in the plot of matlab.
    if(save_mat):
        integral_recorder.save_as_mat("./useful/{}_integral_Controller.mat".format(recorder_version))
        GNE_recorder_no_constraint.save_as_mat("./useful/{}_linear_Controller.mat".format(recorder_version))
        GNE_recorder_constraint.save_as_mat("./useful/{}_linear_Constrained_Controller.mat".format(recorder_version))
        GNE_recorder_constraint_with_hard.save_as_mat("./useful/{}_saturated_linear_Constrained_Controller.mat".format(recorder_version))

    plot_controller_result(integral_recorder,
                GNE_recorder_no_constraint,
                GNE_recorder_constraint,
                GNE_recorder_constraint_with_hard,
                dt=dt)


def main_plot_PSO_GA(pso_ga_version="v_001",
                     save_mat = False):
    if(pso_ga_version is None):
        pso_recorder, ga_recorder = main_run_PSO_GA()

        name_str = "v_pso_ga_001"
        pso_recorder.save("./record/{}_pso_{}.pkl".format(name_str, time_str()))
        ga_recorder.save("./record/{}_ga_{}.pkl".format(name_str, time_str()))

    else:
        path_root = "./useful/"

        recorder_files = [search_file_from_path(path_root, pso_ga_version+"_pso.*pkl"),
                          search_file_from_path(path_root, pso_ga_version+"_ga.*pkl")
                          ]
        pso_recorder = DataRecorder(load_file=recorder_files[0])
        ga_recorder = DataRecorder(load_file=recorder_files[1])

    if(save_mat):
        pso_recorder.save_as_mat("./useful/{}_pso.mat".format(pso_ga_version))
        ga_recorder.save_as_mat("./useful/{}_ga.mat".format(pso_ga_version))
    plot_pso_ga_result(pso_recorder, ga_recorder)


def load_model_200_linear(linear_maddpg_name):
    N = 68
    d = 4
    names = os.listdir("model/"+linear_maddpg_name)
    names.sort()
    names[22*4-1]
    name = "model/"+linear_maddpg_name+"/"+names[22*4-1][:-12]
    env = Env_Closedloop_IEEE_Standard_Bus_Model(N = N, nabla_F=nabla_F_of_game,
                                                      F_of_game=F_of_game)

    device = 'cuda'
    qf_s = Q_Functions_for_MADDPG(env.N, env.obs_num_s, env.action_num_s, env.neighbours, device=device)
    policy = Policy_Linear_Controller(env.N, device="cuda")
    ac_s = ActorCritic_s(env.N, env.obs_num_s, policy, qf_s)

    ac_s.pi_s.load_state_dict(torch.load(name+"pi_s"+".pth"))
    ac_s.q_s.load_state_dict(torch.load(name+"q_s"+".pth"))

    return ac_s.pi_s.K_p[0].detach().cpu().numpy()

def load_model_200_d_4(maddpg_name_d_4):
    N = 68
    d = 4
    names = os.listdir("model/"+maddpg_name_d_4)
    names.sort()
    names[22*4-1]
    name = "model/"+maddpg_name_d_4+"/"+names[22*4-1][:-12]
    env = Env_Closedloop_IEEE_Standard_Bus_Model(N = N, nabla_F=nabla_F_of_game,
                                                      F_of_game=F_of_game)

    device = 'cuda'
    qf_s = Q_Functions_for_MADDPG(env.N, env.obs_num_s, env.action_num_s, env.neighbours, device=device)
    policy = Policy_Piecewise_Controller(env.N, d=d, device=device)
    ac_s = ActorCritic_s(env.N, env.obs_num_s, policy, qf_s)

    ac_s.pi_s.load_state_dict(torch.load(name+"pi_s"+".pth"))
    ac_s.q_s.load_state_dict(torch.load(name+"q_s"+".pth"))

    k_p = ac_s.pi_s.k_plus
    k_m = ac_s.pi_s.k_minus
    b_p = ac_s.pi_s.b_plus
    b_m = ac_s.pi_s.b_minus
    # print([k_p[[0,1,2,3,4,52,53,54], :], k_m[[0,1,2,3,4,52,53,54], :], b_p[[0,1,2,3,4,52,53,54], :], b_m[[0,1,2,3,4,52,53,54], :]])
    print([k_p[0], k_m[0], b_p[0], b_m[0]])
    return k_p[0].detach().cpu().numpy(), k_m[0].detach().cpu().numpy(), b_p[0].detach().cpu().numpy(), b_m[0].detach().cpu().numpy()

def relu(x):
    return np.fmax(x,0)
def monotone_f(x, k_p, k_m, b_p, b_m):

    u_plus_1 = np.sum(k_p*relu(x-b_p), axis = 1, keepdims=True)
    u_plus_2 = -np.sum(k_p[:, 0:-1]*relu(x-b_p[:,1:]),axis = 1, keepdims=True)
            
    u_minus_1 = -np.sum(k_m*relu(b_m-x), axis = 1, keepdims=True)
    u_minus_2 = np.sum(k_m[:, 0:-1]*relu(b_m[:,1:]-x), axis = 1, keepdims=True)
    action = u_plus_1 + u_plus_2 + u_minus_1 + u_minus_2
    
    return action

def plot_function_results(k_p, k_m, b_p, b_m, K_linear):
    N = 1
    x = np.linspace(-2, 2, 100)
    f_s = np.zeros([100, N])
    for i,t in enumerate(x):
        s = np.zeros([N,1])+t
        f_s[i,:]=monotone_f(s, k_p, k_m, 
            b_p, b_m).flatten()
    plt.figure()
    plt.plot(x, f_s[:, 0])
    plt.plot(x, (K_linear*x).flatten())
    plt.xticks([])
    plt.yticks([])
    plt.legend(["Monotonic function", "Linear function"])
    plt.box(False)

    plt.axis([-2, 2, -3, 3])
    plt.savefig("./fig/{}_function.png".format(time_str()))

def main_plot_DSG_MADDPG(maddpg_version = "v_001", save_mat=True):
    if(maddpg_version is None):
        # run_under_d will take over twenty hours
        # run_under_d()
        # These names comes from the runs. 
        # By default, the following runs will be read out
        # or you can assign the names by hand
        dir_names = os.listdir("./model")
        dir_names.sort()
        linear_maddpg_name = dir_names[-11]
        maddpg_name_d_2    = dir_names[-10]
        maddpg_name_d_4    = dir_names[-9]
        maddpg_name_d_8    = dir_names[-8]
        maddpg_name_d_12   = dir_names[-7]
        maddpg_name_d_16   = dir_names[-6]
        maddpg_name_d_20   = dir_names[-5]

        # linear_maddpg_name = "20241206-13-41-45"
        # maddpg_name_d_2    = "20241206-16-20-25"
        # maddpg_name_d_4    = "20241208-15-26-56"
        # maddpg_name_d_8    = "20241206-21-33-35"
        # maddpg_name_d_12   = "20241207-19-46-33"
        # maddpg_name_d_16   = "20241207-19-46-33"
        # maddpg_name_d_20   = "20241207-19-46-33"

        names = [   linear_maddpg_name,
                    maddpg_name_d_2   ,
                    maddpg_name_d_4   ,
                    maddpg_name_d_8   ,
                    maddpg_name_d_12  ,
                    maddpg_name_d_16  ,
                    maddpg_name_d_20    ]
        
        r_s = []
        for name in names:
            r = load_wandb(name="GNE_In_Power_System/"+name,
                                key="test_return")
            r_s.append(r[0:21])

        # linear_maddpg_name = "20241206-13-41-48"
        # maddpg_name_d_4    = "20241208-15-26-59"

        k_p, k_m, b_p, b_m = load_model_200_d_4(maddpg_name_d_4)
        K_linear = load_model_200_linear(linear_maddpg_name)
        maddpg_recorder = DataRecorder()
        for i in range(len(r_s)):
            maddpg_recorder.add(np.array(r_s[i]), "test_return")
        maddpg_recorder.add(np.array(k_p), "k_p")
        maddpg_recorder.add(np.array(k_m), "k_m")
        maddpg_recorder.add(np.array(b_p), "b_p")
        maddpg_recorder.add(np.array(b_m), "b_m")
        maddpg_recorder.add(np.array(K_linear), "K_linear")
        
        name_str = "v_maddpg_001"
        maddpg_recorder.save("./record/{}_{}.pkl".format(name_str, time_str()))
    else:
        path_root = "./useful/"
        recorder_files = [search_file_from_path(path_root, maddpg_version+".*pkl"),
                          ]
        maddpg_recorder = DataRecorder(load_file=recorder_files[0])
    
    r_s = maddpg_recorder.get("test_return")
    plt.figure()
    for i in range(len(r_s)):
        plt.plot(np.linspace(0, 200, 21),r_s[i])
    plt.legend(["linear", "2","4","8","12","16","20"])
    plt.savefig("./fig/{}_test_return.png".format(time_str()))
    k_p = maddpg_recorder.get("k_p")
    k_m = maddpg_recorder.get("k_m")
    b_p = maddpg_recorder.get("b_p")
    b_m = maddpg_recorder.get("b_m")
    K_linear = maddpg_recorder.get("K_linear")
    plot_function_results(k_p, k_m, b_p, b_m, K_linear)

    print("{}".format([k_p, k_m, b_p, b_m, K_linear]))

    if(save_mat):
        maddpg_recorder.save_as_mat("./useful/{}.mat".format(maddpg_version))
    print("plot_MADDPG_over")
    
def load_wandb(name, key):
    api = wandb.Api()
    run = api.run(name)
    m = run.history(samples=1000)
    data = m[key][m[key].notnull()]
    data = data.tolist()
    return np.array(data)
    

if (__name__ == "__main__"):
    init_work_space()

    # main_plot_controller_result_without_RL(recorder_version=None,
                                        #    dt=0.005,
                                        #    save_mat=False)
    # main_plot_PSO_GA(pso_ga_version=None, save_mat=False)
    # main_plot_DSG_MADDPG(maddpg_version=None, save_mat=False)


    # This is used to plot the figures from results that have been stored.
    # NOTE: You should move the files from record dir to useful dir

    main_plot_controller_result_without_RL(recorder_version="v_no_rl_001",
                                           dt=0.005,
                                           save_mat=True)
    main_plot_PSO_GA(pso_ga_version="v_pso_ga_001", save_mat=True)
    main_plot_DSG_MADDPG(maddpg_version="v_maddpg_001", save_mat=True)
    
