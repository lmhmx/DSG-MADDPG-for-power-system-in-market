import scipy

def load_ieee_parameters(N=68):
    if(N==68):
        param = scipy.io.loadmat("ieee-68.mat")
    elif(N==3):
        param = scipy.io.loadmat("ieee-3.mat")
    return param

if(__name__=="__main__"):
    par=load_ieee_parameters()
    print(par)

