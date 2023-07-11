"""
Generates the linear matter power spectrum using CLASS.

Author: Arrykrishna Mootoovaloo
Collaborators: David, Pedro, Jaime
Date: March 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: Emulator for computing the linear matter power spectrum
"""

from dataclasses import dataclass, field
import numpy as np
from classy import Class  # pylint: disable-msg=E0611

# our scripts and functions
import config as CONFIG


def class_compute(cosmology: dict):
    """Pre-computes the quantities in CLASS.

    Args:
        cosmology (dict): A dictionary with the cosmological parameters

    Returns:
        module: A CLASS module
    """

    # instantiate Class
    class_module = Class()

    # set cosmology
    class_module.set(CONFIG.CLASS_ARGS)

    # configuration for neutrino
    class_module.set(CONFIG.NEUTRINO)

    # set basic configurations for Class
    class_module.set(cosmology)

    # compute the important quantities
    class_module.compute()

    return class_module


def delete_module(module):
    """Deletes the module to prevent memory overflow.

    Args:
        module: A CLASS module
    """
    module.struct_cleanup()

    module.empty()

    del module


@dataclass
class PowerSpectrum:
    """Calculates the linear matter power spectrum using CLASS

    Args:
        z_min (float): the minimum redshift
        z_max (float): the maximum redshift
        k_min (float): the minimum wavenumber [unit: 1/Mpc]
        k_max (float): the maximum wavenumber [unit: 1/Mpc]
    """

    # the range of redshifts and wavenumbers to consider
    z_min: float = field(default=0.0)
    z_max: float = field(default=5.0)
    k_min: float = field(default=1e-4, metadata={"unit": "1/Mpc"})
    k_max: float = field(default=1.0, metadata={"unit": "1/Mpc"})

    def __post_init__(self):
        self.wavenumber = np.geomspace(
            self.k_min, self.k_max, CONFIG.NWAVE, endpoint=True
        )

    def pk_linear(self, cosmology: dict, redshift: float = 0.0) -> np.ndarray:
        """Calculates the linear matter power spectrum at a fixed redshift.

        Args:
            cosmology (dict): A dictionary of values following CLASS notation,
            for example, cosmology = {'Omega_b': 0.022, 'Omega_cdm': 0.12,
            'n_s': 1.0, 'h':0.75, 'ln10^{10}A_s': 3.0}
            redshift (float, optional): The redshift at which the power spectrum
            is computed. Defaults to 0.0.

        Returns:
            np.ndarray: The linear matter power spectrum
        """

        # compute the power spectrum
        class_module = class_compute(cosmology)
        pk_values = np.zeros_like(self.wavenumber)
        for i, wav in enumerate(self.wavenumber):
            # get the power spectrum
            pk_values[i] = class_module.pk_lin(wav, redshift)

        # delete the CLASS module
        delete_module(class_module)
        return pk_values

    def pk_nonlinear(self, cosmology: dict, redshift: float = 0.0) -> np.ndarray:
        """
        Calculates the non-linear matter power spectrum at a specific redshift.

        Args:
            cosmology (dict): A dictionary of values following CLASS notation,
            for example, cosmology = {'Omega_b': 0.022, 'Omega_cdm': 0.12,
            'n_s': 1.0, 'h':0.75, 'ln10^{10}A_s': 3.0}
            redshift (float, optional): The redshift at which the power spectrum
            is computed. Defaults to 0.0.

        Returns:
            np.ndarray: The non-linear matter power spectrum.
        """
        # compute the power spectrum
        class_module = class_compute(cosmology)
        pk_values = np.zeros_like(self.wavenumber)
        for i, wav in enumerate(self.wavenumber):
            # get the power spectrum
            pk_values[i] = class_module.pk(wav, redshift)

        # delete the CLASS module
        delete_module(class_module)
        return pk_values
