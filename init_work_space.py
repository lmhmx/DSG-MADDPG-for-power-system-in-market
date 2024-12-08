import os

def init_work_space():
    dir_names = ["data", "fig", "log", "model", "record", "useful"]
    for dir in dir_names:
        if(not os.path.exists(dir)):
            os.mkdir(dir)
            print("Create dir \"{}\"".format(dir))
    
    