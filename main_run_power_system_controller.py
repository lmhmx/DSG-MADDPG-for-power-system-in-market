from IEEE_Standard_Bus_Model import IEEE_Standard_Bus_Model
from DataRecorder import DataRecorder
from PowerSystemController import IntegralController, PiecewiseController
import matplotlib.pyplot as plt
from nabla_F_of_game import nabla_F_of_game
from time_str import time_str, time_int
import numpy as np

def out_scale_Saturated_Linear_Function(u, u_l, u_o, u_ll, u_oo):
    return (u<u_l)*(u_ll-(u_ll-u_l)*np.exp(np.fmin(-(u-u_l)/(u_ll-u_l), 0))) \
            +(u>=u_l)*(u<=u_o)*(u) \
            +(u>u_o)*(u_oo-(u_oo-u_o)*np.exp(np.fmin(-(u-u_o)/(u_oo-u_o), 0)))

def main_run_Integral_Controller():  
    power_model = IEEE_Standard_Bus_Model()
    controller = IntegralController(dt=power_model.dt, N_bus=power_model.N_bus)
    recorder = DataRecorder()

    o = power_model.reset()
    max_time = 20
    it_nums = int(max_time/power_model.dt)
    print("iteration number: {}".format(it_nums))
    it_show_progress = max(int(it_nums/20),1)
    for i in range(it_nums):
        u=controller.output(o)
        o = power_model.step(u)
        theta = o[0:power_model.N_bus]
        omega = o[power_model.N_bus:2*power_model.N_bus]
        recorder.add(omega, "omega")
        recorder.add(theta, "theta")
        recorder.add(o, "o")
        recorder.add(u, "u")
        if(i%it_show_progress==0):
            print("progress: {:6.2f}% {:10.0f}".format(i/it_nums*100, time_int()))

    u_s = recorder.get("u")
    o_s = recorder.get("o")
    omega_s = recorder.get("omega")
    theta_s = recorder.get("theta")
    
    plt_m = 2
    plt_n = 2
    plt.figure(figsize=[plt_n*3, plt_m*3])
    plt.subplot(plt_m, plt_n, 1)
    plt.plot(u_s)
    plt.subplot(plt_m, plt_n, 2)
    plt.plot(u_s[:, [0, 1, 2, 52, 53, 54]])
    plt.subplot(plt_m, plt_n, 3)
    plt.plot(omega_s)
    plt.subplot(plt_m, plt_n, 4)
    plt.plot(omega_s[:, [0, 1, 2, 52, 53, 54]])
    plt.savefig("./fig/"+time_str()+"Integral_Control.png")
    return recorder


def main_run_GNE_Controller(out_scale = np.copy, u_overline = None, u_underline = None):
    power_model = IEEE_Standard_Bus_Model()
    u_overline = 10000*np.ones([power_model.N_bus, 1], 
                               dtype=np.float32) if(u_overline is None) else u_overline
    u_underline = -10000 if(u_underline is None) else u_underline

    controller = PiecewiseController(N=power_model.N_bus, d=15, L=power_model.L, 
                                    dt=power_model.dt, 
                                    nabla_F=nabla_F_of_game, 
                                    P_m=power_model.p,
                                    out_scale=out_scale,
                                    u_overline=u_overline, 
                                    u_underline=u_underline)

    recorder = DataRecorder()

    o=power_model.reset()
    max_time = 20
    it_nums = int(max_time/power_model.dt)
    print("iteration number: {}".format(it_nums))
    it_show_progress = max(int(it_nums/20),1)
    for i in range(it_nums):
        u=controller.output(o)
        o = power_model.step(u)
        theta = o[0:power_model.N_bus]
        omega = o[power_model.N_bus:2*power_model.N_bus]
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
        recorder.add(power_model.dot_omega, "dot_omega")
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

    plt.savefig("./fig/"+time_str()+"GNE_control.png")
    return recorder


def main_run_power_system_controller():
    # Integral controller
    integral_recorder = main_run_Integral_Controller()

    # GNE controller 1
    out_scale = lambda x:x
    u_overline = 10000*np.ones([68, 1], dtype = np.float32)
    u_underline = -u_overline
    GNE_recorder_no_constraint = main_run_GNE_Controller(out_scale=out_scale,
                                     u_overline=u_overline,
                                     u_underline=u_underline)
    
    # GNE controller 2
    out_scale = lambda x:x
    u_overline = 10000*np.ones([68, 1], dtype = np.float32)
    u_overline[[0,1,2],0] = [0.5, 0.4, 0.3]
    u_underline = -u_overline
    GNE_recorder_constraint = main_run_GNE_Controller(out_scale=out_scale,
                                     u_overline=u_overline,
                                     u_underline=u_underline)

    # GNE controller 3
    out_scale = lambda x: out_scale_Saturated_Linear_Function(x, u_l=u_underline, u_o=u_overline,
                                                          u_ll=u_underunderline, u_oo=u_overoverline)

    u_overline = 10000*np.ones([68, 1], dtype = np.float32)
    u_overline[[0,1,2],0] = [0.5, 0.4, 0.3]
    u_overoverline = u_overline + 0.05
    u_underline = -u_overline
    u_underunderline = u_underline - 0.05
    GNE_recorder_constraint_with_hard = main_run_GNE_Controller(out_scale=out_scale,
                                     u_overline=u_overline,
                                     u_underline=u_underline)
    
    return (integral_recorder,
            GNE_recorder_no_constraint,
            GNE_recorder_constraint,
            GNE_recorder_constraint_with_hard)

if(__name__ == "__main__"):
    r1,r2,r3,r4 = main_run_power_system_controller()
    print([r1,r2,r3,r4])
