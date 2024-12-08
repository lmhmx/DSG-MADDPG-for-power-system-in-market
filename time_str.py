import datetime
import time
import sys
import os
import re

def time_int():
    if(not(hasattr(time_int, "is_init"))):
        time_int.is_init = True
        time_int.start_time = time.time()
    return int(time.time()-time_int.start_time)

def time_str():
    return datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')

class Print_Logger(object):
    def __init__(self, filename="./log/log"+time_str()+".txt",
                  show_num = 1,
                  show_in_terminal = True,
                  show_in_file = True):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.num = 0
        self.show_num = show_num
        self.show_in_terminal = show_in_terminal
        self.show_in_file = show_in_file
 
    def write(self, message):
        if(self.num%self.show_num==0):
            if(self.show_in_terminal):
                self.terminal.write(message)
            if(self.show_in_file):
                self.log.write(message)
        self.num += 1
    def flush(self):
        pass
class Counter:
    def __init__(self):
        self._num = 0
    @property
    def count(self):
        self._num+=1
        return self._num

def search_file_from_path(path:str, search_target):
    dirs = os.listdir(path)
    result = None
    for dir in dirs:
        # If we fine such a file
        if(re.match(search_target, dir) is not None):
            result = dir
            break
    else:
        # We cannot find such a file
        print("Warning: in the path '{}', there is not a file name started by '{}'".format(path, search_target))
        return ""
    return os.path.join(path, result)
if(__name__=="__main__"):
    print("current time: {}".format(time_str()))
