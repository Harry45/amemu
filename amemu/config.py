"""
The main setting file for the emulator

Author: Arrykrishna Mootoovaloo
Collaborators: David, Pedro, Jaime
Date: March 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: Emulator for computing the linear matter power spectrum
"""

# number of wavenumber to consider [equal to the number of GPs]
NWAVE = 40

# minimum and maximum redshift and wavenumber [1/Mpc]
Z_MIN = 0.0
Z_MAX = 5.0
K_MIN = 1e-4
K_MAX = 50.0

# fixed quantities for the power spectrum (k_pivot is from CLASS)
FIX_K_PIVOT = 0.05
FIX_SIGMA8 = 0.80
FIX_NS = 1.0


# CLASS output
CLASS_ARGS = {
    "output": "mPk",
    "P_k_max_1/Mpc": K_MAX,
    "z_max_pk": Z_MAX,
    "non linear": "halofit",
}

# neutrino settings
NEUTRINO = {"N_ncdm": 1.0, "deg_ncdm": 3.0, "T_ncdm": 0.71611, "N_ur": 0.00641}

# cosmological parameters (DES) - we include redshift as an input
COSMO = ["Omega_cdm", "Omega_b", "h", "z"]

# priors (DES)
PRIORS = {
    "Omega_cdm": {"distribution": "uniform", "specs": [0.07, 0.43]},
    "Omega_b": {"distribution": "uniform", "specs": [0.028, 0.027]},
    "h": {"distribution": "uniform", "specs": [0.64, 0.18]},
    "z": {"distribution": "uniform", "specs": [Z_MIN, Z_MAX]},
}

# the Gaussian Process settings
LEARN_RATE = 1e-2
NRESTART = 2
NITER = 1000

# URL where the models are stored
MODEL_LINK = "http://dl.dropboxusercontent.com/s/2yojoa9wcn2m5wk/emulin-gps.tar?dl=0"
