from Env_IEEE_Standard_Bus_Model import Env_IEEE_Standard_Bus_Model
from DataRecorder import DataRecorder
from PowerSystemController import IntegralController
import matplotlib.pyplot as plt
from time_str import time_str

def main_run_Integral_Controller():  
    env = Env_IEEE_Standard_Bus_Model()
    controller = IntegralController(dt=env.power_model.dt, N_bus=env.power_model.N_bus)
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
        if(i%it_show_progress==0):
            print("progress: {:3.2f}%".format(i/it_nums*100))

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
    plt.savefig("./fig/"+time_str()+".png")
    return recorder

if(__name__=="__main__"):
    main_run_Integral_Controller()
