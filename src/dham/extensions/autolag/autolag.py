# Required package imports
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Standard Plot Settings
plt.rcParams["figure.figsize"] = (11.7,8.3)
STANDARD_SIZE=24
SMALL_SIZE=18
plt.rc('font', size=STANDARD_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=STANDARD_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=STANDARD_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

# src Imports