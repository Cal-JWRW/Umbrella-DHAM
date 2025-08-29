# Required package imports
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
from read_px import read_px
from read_log import read_log
from generate_count_matrix import gcm
from generate_mm import gen_mm
from mm2pmf import mm2pmf
from extensions.autocut.autocut import autocut
from extensions.autodis.autodis import autodis

R = 0.008314 # universal gas constant

def dham(il, ix):

    # Make direcotry to store results
    results_dir = Path("DHAM_Results")
    results_dir.mkdir(exist_ok=True)

    with open(il, 'r') as f:
        log_files = [line.rstrip() for line in f]

    with open(ix, 'r') as f:
        px_files = [line.rstrip() for line in f]

    # Extract simulation trajectories
    u_data = np.array([read_px(f) for f in tqdm(px_files, desc='Extracting trajectories ...')], dtype=np.float64)

    # Take the absolute of the reaction coordinate and convert to angstrom
    u_data = np.abs(u_data) * 10

    # Load in the US log files to extract umbrella centers, biasing potentials and the temperature
    log_data = np.array([read_log(f) for f in tqdm(log_files, desc='Extracting umbrella centers, biasing constants and temperature...')])

    # Assume all simulations are the same temperature (AS THEY SHOULD BE) so only extract one
    temp = log_data[0,0]

    # Extract umbrella centers
    uc = log_data[:, 1]

    # Take the absolute of the umbrella cenetrs and convert to angstrom
    uc = np.abs(uc) * 10

    # Extract biasing potentials
    bk = log_data[:, 2]

    # Convert bias potentials into kT 
    bk = bk/(temp*R*100)

    # Calculate how much of the initial trajectory should be discarded
    equilibration_point = autocut(u_data)

    # Remove equilibration portion from data
    u_data = u_data[:, equilibration_point:]

    # Determine the ideal discretization
    #n_bins = autodis(u_data, uc, bk, temp, results_dir / 'Discretization.png')
    n_bins=400
    # Calculate umbrella sampling histograms

    # Extract minimum and maximum of the trajectories
    trj_min = np.min(u_data)
    trj_max = np.max(u_data)

    # Calculate bins
    bins = np.linspace(trj_min, trj_max, n_bins)

    # Discretize trajectories
    print('Discretizing Trajectories...')
    discretized_trajectory = np.digitize(u_data, bins)-1 
    print('Discretizion Complete!')

    # Empty array to hold histogram counts
    HCount = np.zeros(len(bins))

    # Populate histograms
    for k in tqdm(discretized_trajectory, desc='Populating US Histograms...'):
        for i in k:
            HCount[i] += 1

    # Plot histograms
    fig, ax = plt.subplots()
    ax.fill_between(bins, HCount, np.zeros(len(bins)))
    ax.set_xlabel(r'$z$ / $\rm{}\AA$')
    ax.set_ylabel(r'$Histogram \: Count$')

    # Save histograms
    fig.savefig(results_dir / "Histograms.png", dpi=300)

    return

if __name__ == '__main__':
    
    import argparse 

    whoami = "WRITE ME"
    parser = argparse.ArgumentParser(description = whoami, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-il', type=str, nargs=1, required= True, help=' .dat file containing ordered names of GROMACS log file')
    parser.add_argument('-ix', type=str, nargs=1, required= True, help=' .dat file containing ordered names of GROMACS pullx files')

    args = parser.parse_args()

    dham(
        args.il[0],
        args.ix[0]
    )