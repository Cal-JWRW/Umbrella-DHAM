# Required package imports
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

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
from generate_count_matrix import gcm
from generate_mm import gen_mm

def autolag(d_traj, bin_edges, uc, bk, result_path):

    # Calculate the number of bins from the bin-edges
    n = len(bin_edges)

    # Generate test lagtimes
    lagtimes = np.arange(10, 5010, 100)

    # Empty array to hold relaxation times
    rt = np.zeros(len(lagtimes))
    
    # Iterate over lagtimes, construct MM and calculate the implied relaxation timescales
    for it in tqdm(range(len(lagtimes)), desc='Calculating Implied Timescales...'):

        lt = lagtimes[it]

        transition_matrix, transition_sums = gcm(d_traj, n, lt) # Construct transition matrix at current lagtime

        mm = gen_mm(transition_matrix, transition_sums, bin_edges, uc, bk) # Construct the resulting mm

        eigenvalues, _ = eig(np.transpose(mm))

        # Sort eigenvalues by magnitude, descending
        sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]

        # The second largest eigenvalue (by magnitude)
        lambda2 = sorted_eigs[1]

        rt[it] = -(lt/np.log(lambda2))


    logrt = np.log10(rt)

    x_diff = np.diff(lagtimes)
    y_diff = np.diff(rt)

    slope = y_diff/x_diff

    norm_slope = slope/np.max(slope)

    acceptable_lags = np.where(norm_slope <= 0.01)[0]+1

    lagtime = lagtimes[acceptable_lags[0]]
    
    fig, ax = plt.subplots()

    ax.plot(lagtimes, logrt, lw=3.0, label='Implied Relaxation Timescales')

    ax.axvline(lagtime, linestyle='--', lw=3.0, label='Markov Timescale')

    ax.set_xlabel(r'$Lagtime$ / $\rm{} Timesteps$')
    ax.set_ylabel(r'$log\;(\;Implied \: Relaxation \: Timescale\;)$ / $\rm{} Timesteps$')

    ax.set_ylim((0, 12))

    fig.savefig(result_path / "Lagtime_Optimization.png", dpi=300)

    data = np.column_stack((lagtimes, logrt))

    # === Prepare Header for Output File ===
    full_header = '@    title "Markov Model Implied Relaxation Timescales"' + '\n'
    full_header += '@    xaxis  label "Lagtime / Timestep"' + '\n'
    full_header += '@    yaxis  label "log(Implied Relaxation Timescale) / Timestep"' + '\n'
    full_header += '@    s0 legend "Implied Relaxation Timescale" '

    # === Save Results to File ===
    np.savetxt(
        result_path / "Lagtime_Optimization.xvg",  # Output file path
        data,                    # Transpose to get 2 columns: time, tilt
        fmt="%.2f",                  # Format numbers to 2 decimal places
        delimiter="\t",              # Tab-separated values
        header=full_header,         # Header describing the file
        comments=''                 # No comment character before header lines
    )



    return lagtime

