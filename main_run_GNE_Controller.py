from Env_IEEE_Standard_Bus_Model import Env_IEEE_Standard_Bus_Model
from DataRecorder import DataRecorder
from PowerSystemController import IntegralController
import matplotlib.pyplot as plt
from time_str import time_str
from Env_IEEE_Standard_Bus_Model import Env_IEEE_Standard_Bus_Model
from DataRecorder import DataRecorder
from PowerSystemController import PiecewiseController
import matplotlib.pyplot as plt
from nabla_F_of_game import nabla_F_of_game
from time_str import time_str, time_int
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_gain", type=float, default=1.0)
    args = parser.parse_args()
    args.out_scale = lambda x:args.out_gain*x
    return args

def main_run_GNE_Controller(out_scale = np.copy, u_overline = None, u_underline = None):
    env = Env_IEEE_Standard_Bus_Model()
    u_overline = 10000*np.ones([env.N_bus, 1], 
                               dtype=np.float32) if(u_overline is None) else u_overline
    u_underline = -10000 if(u_underline is None) else u_underline

    controller = PiecewiseController(N=env.N_bus, d=15, L=env.power_model.L, 
                                    dt=env.power_model.dt, 
                                    nabla_F=nabla_F_of_game, 
                                    P_m=env.power_model.p,
                                    out_scale=out_scale,
                                    u_overline=u_overline, 
                                    u_underline=u_underline)

    recorder = DataRecorder()

    o, _=env.reset()
    max_time = 20
    it_nums = int(max_time/env.power_model.dt)
    print("iteration number: {}".format(it_nums))
    it_show_progress = max(int(it_nums/20),1)
    for i in range(it_nums):
        u=controller.output(o)
        o ,_ , _, _, _ = env.step(u)
        theta = o[0:env.power_model.N_bus]
        omega = o[env.power_model.N_bus:2*env.power_model.N_bus]
        recorder.add(omega, "omega")
        recorder.add(theta, "theta")
        recorder.add(o, "o")
        recorder.add(u, "u")
        recorder.add(controller.lambda_, "lambda")
        recorder.add(controller.mu_, "mu")
        recorder.add(controller.s, "s")
        recorder.add(controller.dot_s, "dot_s")
        recorder.add(controller.nu_minus, "nu_minus")
        recorder.add(controller.nu_plus, "nu_plus")
        recorder.add(env.power_model.dot_omega, "dot_omega")
        recorder.add(controller.dot_u, "dot_u")

        if(i%it_show_progress==0):
            print("progress: {:6.2f}% {:10.0f}".format(i/it_nums*100, time_int()))

    u_s = recorder.get("u")
    o_s = recorder.get("o")
    omega_s = recorder.get("omega")
    theta_s = recorder.get("theta")
    lambda_s = recorder.get("lambda")
    mu_s = recorder.get("mu")
    s_s = recorder.get("s")
    dot_s = recorder.get("dot_s")
    nu_minus_s = recorder.get("nu_minus")
    nu_plus_s = recorder.get("nu_plus")
    dot_omega = recorder.get("dot_omega")
    dot_u = recorder.get("dot_u")
    plt_m = 4
    plt_n = 4
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
    plt.ylabel("omega 52 68")
    plt.plot(omega_s[:, 52:68])

    plt.subplot(plt_m,plt_n,10)
    plt.plot(dot_s)
    plt.ylabel("dot_s")

    plt.subplot(plt_m,plt_n,11)
    plt.plot(dot_s[:, 52:68])
    plt.ylabel("dot_s 52-68")

    plt.subplot(plt_m,plt_n,12)
    plt.plot(dot_omega)
    plt.ylabel("dot_omega")

    plt.subplot(plt_m,plt_n,13)
    plt.plot(dot_omega[:, 52:68])
    plt.ylabel("dot_omega 52-68")

    plt.subplot(plt_m,plt_n,14)
    plt.plot(dot_u)
    plt.ylabel("dot_u")

    plt.subplot(plt_m,plt_n,15)
    plt.plot(dot_u[:, 52:68])
    plt.ylabel("dot_u 52-68")

    plt.savefig("./fig/"+time_str()+".png")
    return recorder

if(__name__ == "__main__"):
    args = get_args()
    u_overline = 10000*np.ones([68, 1], dtype = np.float32)
    # u_overline[[0,1,2],0] = [0.5, 0.4, 0.3]
    u_underline = -u_overline
    record = main_run_GNE_Controller(out_scale=args.out_scale,
                                     u_overline=u_overline,
                                     u_underline=u_underline)
    record.save("./record/v_001_linear_Controller_{}.pkl".format(time_str()))
    # record.save("./record/v_001_linear_Constrained_Controller_{}.pkl".format(time_str()))

