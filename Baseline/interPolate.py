from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import math

class calinterPlate:
    def __init__(self,CpArray):
        self.func = interploate(CpArray)
        self.CpArray = CpArray
        self.x_bigen = CpArray[np.argmin(CpArray[:,0]),0]
        self.x_end = min(CpArray[0,0],CpArray[-1,0])

    def compute_argmin(self):
        # compute the argmin of the CpArray
        x = np.linspace(self.x_bigen,self.x_end,100)
        id_min = np.argmin(self.func[1](x))
        argmin = x[id_min]
        return argmin

    def compute_min(self):
        # compute the global min and middle min of the CpArray
        x_global = np.linspace(self.x_bigen,self.x_end,100)
        x_middle = np.linspace(0.2,0.8,60)
        Cpmin_global = np.min(self.func[1](x_global))
        Cpmin_middle = np.min(self.func[1](x_middle))
        Cpmin = [Cpmin_global, Cpmin_middle]
        return Cpmin
    
    def compute_middle_parameter(self):
        # compute the k_fit of the CpArray
        x = np.linspace(0.2,0.8,100)
        Cp_distance = self.func[0](x) - self.func[1](x)
        k, _ = np.polyfit(x, Cp_distance, 1)
        parameter = k
        return parameter

    def is_injected(self):
        # compute the degree of crossover
        x = np.linspace(self.x_bigen,0.2,100)
        diff = self.func[0](x) - self.func[1](x)
        value_is_injected = np.min(diff)
        return value_is_injected
    

def interploate(CpArray):
    # interpolate the CpArray given by the panel method
    x = CpArray[:,0]
    Cp_array_up = CpArray[0:np.argmin(x)+1,:]
    Cp_array_down = CpArray[np.argmin(x):,:]

    x_up = Cp_array_up[:,0]
    Cp_up = Cp_array_up[:,1]

    x_down = Cp_array_down[:,0]
    Cp_down = Cp_array_down[:,1]

    f_up = interp1d(x_up,Cp_up) # function of the upper edge
    f_down = interp1d(x_down,Cp_down) # function of the lowwer edge
    func = [f_up, f_down]
    return func