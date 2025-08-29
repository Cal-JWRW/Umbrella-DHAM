import numpy as np
from scipy.linalg import eig

def mm2pmf(mm, T):
    """
    Convert a transition matrix (or similar square matrix) into a
    potential of mean force (PMF).
    
    Steps:
    1. Compute the dominant eigenvector of the matrix transpose.
    2. Normalize it into a probability distribution.
    3. Convert probabilities into free energies (PMF) using -log(p).
    4. Shift PMF so the minimum value is zero.
    """

    # Compute eigenvalues and eigenvectors of the transposed matrix
    eigval, eigvec = eig(np.transpose(mm))

    # Select the eigenvector corresponding to the largest eigenvalue
    prob = eigvec[:, np.where(eigval == np.max(eigval))[0][0]]

    # Normalize eigenvector so it sums to 1 → valid probability distribution
    prob = prob / np.sum(prob)

    # Convert probability distribution to potential of mean force (PMF)
    pmf = -np.log(prob)

    # Shift PMF so the minimum energy level is zero (relative scale)
    pmf -= np.min(pmf)

    return pmf