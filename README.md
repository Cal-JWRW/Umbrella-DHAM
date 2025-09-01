# Umbrella-DHAM
This repository contains a python implementation of the Dynamic Histogram Analysis Method (DHAM) [1] for the calculation of free energy profiles and permeation coefficients from lipid membrane umbrella sampling simulations. Our implementation is designed to be easy to use, and take similar input to the in-built GROMACS WHAM tool. DHAM utilizes a Markov model for it's free energy estimates, such models typically require parameterisation with regards to the amount of bins are used in the model and the lagtime (&#120591;). 

Our implementation selects an optimal value for each of these as well as the amount of sampling time to discard due to "burn in" or "equilibration". 

# Bin Optimization
Accurate estimation of free energy profiles using Markov models requires discretization of the state space that is fine enough to capture relevant transitions, but not so fine that statistical noise or numerical artifacts dominate. Since the optimal discretization is generally unknown a priori, we determine it empirically. Markov models are constructed at progressively finer levels of discretization, and the corresponding free energy profiles are estimated. To monitor convergence, we compute the integral of each free energy profile. As discretization becomes sufficiently fine, the integral plateaus, indicating that further refinement does not significantly alter the free energy landscape. We quantify this plateau by calculating the normalized absolute gradient of the integral with respect to bin number and identify regions where the gradient falls below a small threshold. The midpoint of this plateau is selected as the discretization level for constructing a finely resolved Markov model, ensuring both numerical stability and physical fidelity in the estimated free energy profile.

# Lag time Optimization
Since Markov models are constructed from time-series data, a lagtime (τ) must be chosen to ensure transitions are approximately Markovian, i.e., memoryless. The lagtime must be long enough that future states depend only on the current state, but short enough to capture the relevant kinetics of the system. As with discretization, the optimal lagtime cannot be known a priori and is determined empirically. We construct Markov models over a range of lagtimes (10–5000 timesteps) and compute the eigenvalues and eigenvectors of the transition matrices. Using the first nontrivial eigenvector, we calculate the implied relaxation timescale of the slowest process. As τ increases, this timescale plateaus, indicating that Markovian behavior is achieved. We quantify the plateau by calculating the normalized absolute gradient of the relaxation timescale with respect to lagtime and select the first lagtime with a gradient ≤ 0.02. This lagtime is adopted as the optimal Markov timescale, balancing Markovian validity and kinetic relevance.

# Automatic Equilibration Point Detection
It is standard practice to discard an initial portion of umbrella sampling data as “equilibration” or “burn-in” to ensure sampling from a steady-state distribution and to remove bias from the starting configurations. Rather than selecting this period arbitrarily, we implement an automated procedure to determine the burn-in time. For each umbrella sampling trajectory, we calculate the normalized autocorrelation function. The burn-in period is defined as the first timestep at which the autocorrelation decays to near zero (within statistical noise), and the longest such decorrelation time across all trajectories is used as the burn-in for the model. This approach ensures that the retained data are statistically decorrelated from the initial configurations. 

# Usage

The tool works similarly to the GROMACS WHAM tool. From a directory containing the umbrella sampling pullx and log files, construct a mirrored list of log files stored in .dat files:

ls -1v *.log > log.dat
ls -1v * pullx * > px.dat

Following this, the DHAM tool can be called from src/DHAM.py with:
python /path/to/package/DHAM.py -il log.dat -ix px.dat

# Limitations
This implementation is built to take GROMACS output files and does not curretly accept output files from other simulation packages, however, we intend to address this in future releases.



# References
1. Rosta, E. and Hummer, G. (2014) ‘Free energies from dynamic weighted histogram analysis using unbiased Markov State model’, Journal of Chemical Theory and Computation, 11(1), pp. 276–285. doi:10.1021/ct500719p. 
