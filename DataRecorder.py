import numpy as np
import pickle
from scipy.io import savemat
class DataRecorder:
    def __init__(self, names=[], lengths=[], max_size=15, load_file = None) -> None:
        """
        names: the names of the variables
        lengths: the length of the variables
        max_size: the pre-allocated size of the variables
            - when it is a number, all variables allocate the same size
            - when it is a list, it assign different size for the variables  
        load_file: the file to be loaded
            - when it is None, do not load
            - when it is not None, excute self.load()
        """
        if(load_file is None):
            self.data = {}
            self.max_sizes = {}
            self.current_size = {}
            self.add_names(names, lengths, max_size)
        else:
            self.load(load_file)
    def add_names(self, names, lengths=[], max_size=15):
        """
        names: the names of the variables
        lengths: the length of the variables
        max_size: the pre-allocated size of the variables
            - when it is a number, all variables allocate the same size
            - when it is a list, it assign different size for the variables  
        """
        max_sizes = None
        if (type(max_size)==int):
            max_sizes = [max_size]*len(names)
        else:
            max_sizes = max_size
        for i in range(len(names)):
            if(lengths==[]):
                self.add_name(names[i], -1, max_sizes[i])
            else:
                self.add_name(names[i], lengths[i], max_sizes[i])
            
    def add_name(self, name, length=-1, max_size=15):
        """
        name: str
        length: int
        max_size: int
        """
        if(length!=-1):
            self.data[name]=np.zeros([max_size, length], dtype=np.float32)
            self.max_sizes[name] = max_size
            self.current_size[name] = 0

    def add(self, x: np.ndarray, name):
        """
        x: vector
        name: str
        """
        if(name not in self.data.keys()):
            self.add_name(name, length=len(x.flatten()), max_size=15)
        if(self.current_size[name]==self.max_sizes[name]):
            tmp = self.data[name]
            self.data[name] = np.zeros([np.shape(self.data[name])[0]*2, np.shape(self.data[name])[1]], dtype=np.float32)
            self.data[name][0:self.max_sizes[name], :] = tmp
            self.max_sizes[name] *= 2
            del tmp
        self.data[name][self.current_size[name], :] = x.flatten()
        self.current_size[name] += 1
    def get(self, name):
        return self.data[name][0:self.current_size[name], :]
    def clear(self):
        del self.data
        # for key in self.data.keys():
        #     del self.data[key]
    def save(self, path="./record/recorder.pkl"):
        with open(path,'wb') as file:
            pickle.dump({"data":self.data, 
                         "max_sizes":self.max_sizes,
                         "current_size":self.current_size},file)
    def load(self, path = "./record/recorder.pkl"):
        with open(path,'rb') as file:
            data = pickle.load(file)
            self.data=data["data"]
            self.max_sizes=data["max_sizes"]
            self.current_size = data["current_size"]
    def save_as_mat(self, path="./record/recorder.mat"):
        """
        This function works for outputing data accessed by MATLAB, which is only used when plotting.
        Be sure to not use it anytime else.
        """
        savemat(path, {k:self.data[k][0:self.current_size[k]] for k in self.data.keys()})

if(__name__=="__main__"):
    data_recorder = DataRecorder()
    data_recorder.load()
