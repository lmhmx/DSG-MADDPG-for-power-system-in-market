import sys
import os
import random
def run_under_d():
    d_s=[2, 4, 8, 12, 16, 20]
    # d_s = [4, 16, 20]

    code_args = "--use_cmd_args y"+\
                    " --device cuda --gamma 0.999 --polyak 0.95"+\
                    " --lr 0.005 --batch_size 8192 --epoch 252" + \
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
                    " --use_linear_policy y" +\
                    " --replay_buffer_size 90000"
    cmd_str = "python run_MADDDPG.py " + code_args
        

    ans=os.system(cmd_str)

    for d in d_s:
        code_args = "--use_cmd_args y"+\
                    " --device cuda --gamma 0.999 --polyak 0.95"+\
                    " --lr 0.005 --batch_size 8192 --epoch 252" + \
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
                    " --replay_buffer_size 90000"
                    # " --d 10"
        cmd_str = "python run_MADDDPG.py " + code_args
        
        cmd_all_str=cmd_str + " --d {}".format(d)
        ans=os.system(cmd_all_str)
        if(ans!=0): # 若出错，则判断
            a=input("continue? y/n    ")
            if(a=="n"):
                break
if(__name__ == "__main__"):
    run_under_d()
