from numba import njit
import numpy as np

@njit
def gcm(disc_data, nbins, lagt):
    """
    Compute transition count matrices from discretized trajectories.

    Parameters
    ----------
    disc_data : ndarray, shape (ntraj, nsteps)
        Array of discretized trajectories.
    nbins : int
        Number of discrete states (bins).
    lagt : int
        Lag time between current and next state when counting transitions.

    Returns
    -------
    transition_matrix : ndarray, shape (nbins, nbins)
        Global transition count matrix (summed over all trajectories).
    transition_total : ndarray, shape (ntraj, nbins)
        For each trajectory, the number of transitions from each state.
    """

    # Number of trajectories and number of time steps per trajectory
    ntraj, nsteps = disc_data.shape

    # Transition count matrices per trajectory: nCm[i] is nbins x nbins
    nCm = np.zeros((ntraj, nbins, nbins), dtype=np.int64)

    # Loop over all trajectories
    for k in range(ntraj):
        # Loop over time steps, starting from lag time
        for i in range(lagt, nsteps):

            # State at time t
            current_state = disc_data[k, i-lagt]

            # State at time t + lag
            transition_state = disc_data[k, i]

            # Increment the transition count for this trajectory
            nCm[k, current_state, transition_state] += 1

    # Global transition matrix (sum over all trajectories)
    transition_matrix = np.zeros((nbins, nbins), dtype=np.int64)

    # Per-trajectory totals: how many transitions leave each state
    transition_total = np.zeros((ntraj, nbins), dtype=np.int64)

    for k in range(ntraj):
        for i in range(nbins):
            # Add trajectory-specific transitions to the global matrix
            for j in range(nbins):
                transition_matrix[i, j] += nCm[k, i, j]

            # Count total outgoing transitions from state i in trajectory i
            transition_total[k, i] = np.sum(nCm[k, i, :])

    return transition_matrix, transition_total