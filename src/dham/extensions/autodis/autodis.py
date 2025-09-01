import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Package Imports
from generate_count_matrix import gcm
from generate_mm import gen_mm
from mm2pmf import mm2pmf

def trapz(x, y):
    return(np.trapz(y, x))


def autodis(u_data, uc, bk, temp, result_path):

    integrals = []
    n_bins = []

    trj_min = np.min(u_data)
    trj_max = np.max(u_data)

    for n in tqdm(range(100, 1000, 50), desc='Performing discretization optimization...'):

        n_bins.append(n)

        bin_edges = np.linspace(trj_min, trj_max, n)
     

        d_traj = np.digitize(u_data, bin_edges) - 1

        transition_matrix, transition_sums = gcm(d_traj, n, 1500) # Semi-large initial lagtime guess 

        mm = gen_mm(transition_matrix, transition_sums, bin_edges, uc, bk)
    
        try:
            pmf = mm2pmf(mm, temp)
        except ValueError:
            integrals.append(np.nan)
            continue
        
        pmf = pmf * temp * 0.008314
        valid_mask = (bin_edges > np.min(uc)) & (bin_edges < np.max(uc))

        valid_bin = bin_edges[valid_mask]
        valid_pmf = pmf[valid_mask]

        integrals.append(trapz(valid_bin, valid_pmf))

    integrals = np.array(integrals)
    n_bins = np.array(n_bins)

    # Remove nan entries that occur as a result of over-discretization
    nan_mask = ~np.isnan(integrals)
    integrals = integrals[nan_mask]
    n_bins = n_bins[nan_mask]

    dy = np.diff(integrals)
    dx = np.diff(n_bins)
    slope = abs(dy/dx)

    n_slope = slope/np.max(slope)

    below_thresh = np.where(n_slope <= 0.1)[0]

    ideal_discretization = n_bins[below_thresh[len(below_thresh)//2]]

    fig, ax = plt.subplots()

    ax.plot(n_bins, integrals, lw=3.0)
    ax.axvline(ideal_discretization, lw=3.0, linestyle='--', color='black', label='Ideal bin number')

    ax.set_xlabel(r'$Number \: of \: Bins$')
    ax.set_ylabel(r'$\int_{}^{}\Delta G $')
    
    ax.legend()

    plt.tight_layout()

    fig.savefig(result_path / 'Discretization.png', dpi=300)

    data = np.column_stack((n_bins, integrals))

    data = data.astype(np.float64)

    # === Prepare Header for Output File ===
    full_header = '@    title "Markov Model Discretization optimization"' + '\n'
    full_header += '@    xaxis  label "n Bins"' + '\n'
    full_header += '@    yaxis  label "Area Under Free Energy"' + '\n'
    full_header += '@    s0 legend "Area Under Free Energy" '

    # === Save Results to File ===
    np.savetxt(
        result_path / "Discretization_Optimization.xvg",  # Output file path
        data,                    # Transpose to get 2 columns: time, tilt
        fmt="%.2f",                  # Format numbers to 2 decimal places
        delimiter="\t",              # Tab-separated values
        header=full_header,         # Header describing the file
        comments=''                 # No comment character before header lines
    )

    return ideal_discretization
    


