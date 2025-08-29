import numpy as np
import scipy.signal as sig
from tqdm import tqdm

def autocut(trajectories, thresh=0.01):
    """
    Determines the longest decorrelation time across multiple trajectories.

    Parameters:
    -----------
    trajectories : list of np.array
        Each element is a time series (trajectory) to analyze.
    thresh : float
        Threshold for determining decorrelation (default 0.01).

    Returns:
    --------
    trajectory_cut : int
        Maximum index where autocorrelation falls below threshold across all trajectories.
    """

    trajectory_cut = 0   # Initialize maximum decorrelation index
    deco_points = []     # Store decorrelation indices for each trajectory (optional)

    for traj in tqdm(trajectories, desc='Computing equilibration cut-off...'):
        # Center the trajectory by subtracting the mean
        tm = traj - np.mean(traj)

        # Compute autocorrelation of the centered trajectory
        autocorr = sig.correlate(tm, tm)

        # Normalize so that autocorrelation at lag 0 is 1
        autocorr = autocorr / (np.var(tm) * len(tm))

        # Take only the non-negative lags
        autocorr = autocorr[autocorr.size // 2:]

        # Find indices where autocorrelation falls below the threshold
        mask = (autocorr <= thresh)
        cut_indices = np.where(mask)[0][0]  # First index below threshold

        # Update the maximum decorrelation index if this trajectory is longer
        if cut_indices >= trajectory_cut:
            trajectory_cut = cut_indices

        # Store the individual trajectory's decorrelation index
        deco_points.append(cut_indices)

    # Return the largest decorrelation index across all trajectories
    return trajectory_cut
