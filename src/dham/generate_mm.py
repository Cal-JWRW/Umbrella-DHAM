import numpy as np
from numba import njit

@njit
def gen_mm(transition_matrix, transition_total, bin_edges, uc, bias):
    """
    Generate a reweighted Markov transition matrix from biased simulation data.
    
    Parameters
    ----------
    transition_matrix : 2D array
        Matrix of observed transition counts between states (i → j).
    transition_total : 2D array
        Total transitions per umbrella window (indexed by [window, state]).
    bin_edges : 1D array
        Discretized coordinate bin edges.
    uc : 1D array
        Umbrella centers (bias potentials are centered here).
    bias : 1D array
        Bias strengths (spring constants) for each umbrella window.
    
    Returns
    -------
    mm : 2D array
        Reweighted and normalized Markov transition probability matrix.
    """

    # Allocate space for the reweighted transition matrix
    mm = np.empty(shape=transition_matrix.shape, dtype=np.longdouble)

    # Compute the spacing between consecutive bins (assumes uniform binning)
    bin_spacing = bin_edges[1] - bin_edges[0]

    # Loop over all pairs of states (i, j)
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[0]):

            # Only process if at least one transition from i → j was observed
            if transition_matrix[i, j] > 0:

                # Initialize denominator for reweighting factor
                denom = 0.0

                # Loop over all umbrella windows (biased simulations)
                for k in range(transition_total.shape[0]):

                    # Compute bias potential for each bin:
                    # u(x) = (1/2) * k_bias * (x_center - bin_edges)^2
                    # shift bins by half spacing to get bin centers
                    u = (0.5 * bias[k]) * np.square(uc[k] - bin_edges - bin_spacing/2)

                    # Only include window k if state i was visited there
                    if transition_total[k, i] > 0:

                        # Add contribution to denominator:
                        # weighted by number of visits and exponential bias correction
                        denom += transition_total[k, i] * np.exp(-(u[j] - u[i]) / 2)

                # Reweight transition count → unbiased contribution
                mm[i, j] = transition_matrix[i, j] / denom

    # Normalize each row so rows sum to 1 → valid Markov transition matrix
    mm = mm / np.sum(mm, axis=1)[:, None]

    return mm
