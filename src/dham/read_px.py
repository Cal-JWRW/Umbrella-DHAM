import numpy as np

def read_px(filename):

    data = np.loadtxt(filename, skiprows=17)[:,1]

    return data



