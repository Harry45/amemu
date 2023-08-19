"""
Author: Dr. Arrykrishna Mootoovaloo
Date: July 2023
Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
Description: Prediction of the linear matter power spectrum using the GP emulator
"""
import os
from typing import Tuple, Union
from pathlib import Path
import urllib.request
import shutil
import tarfile
import torch
import numpy as np

# our scripts and functions
from amemu.utils.helpers import load_list, MyProgressBar
from amemu.trainingpoints import generate_prior
import amemu.config as CONFIG


def download_pre_trained_model(link: str):
    """
    Download all pre-trained models from a given link.

    Args:
        link (str): the link where the models are stored.
    """
    parent_path = Path(__file__).parents[0]
    emulator_name = "linear-pk-emulator.tar"
    urllib.request.urlretrieve(link, emulator_name, MyProgressBar())
    tar_file = tarfile.open(emulator_name, "r")
    tar_file.extractall(path=parent_path)
    tar_file.close()
    if os.path.isfile(emulator_name):
        os.remove(emulator_name)


def load_gps(nlhs: int = 500, download_link: str = None, download: bool = True) -> list:
    """
    Load the pre-trained GPs based on the number of LH used.

    Args:
        nlhs (int, optional): The number of LH points. Defaults to 500.
        download_link (str): link where the GPs are stored

    Returns:
        list: a list of GPs.
    """

    gps = []
    parent_path = Path(__file__).parents[0]
    gp_path = os.path.join(parent_path, f"gps/{nlhs}")

    if download:
        if os.path.exists(gp_path):
            print("Removing old pre-trained models.")
            shutil.rmtree(gp_path)
        print("Now downloading latest pre-trained models.")
        download_pre_trained_model(download_link)

    for i in range(CONFIG.NWAVE):
        gp_module = load_list(gp_path, f"pk_linear_lhs_{nlhs}_wave_{i}")
        gps.append(gp_module)
    return gps


def generate_cosmo_prior() -> dict:
    """
    Generates the prior for the cosmological parameters.

    Returns:
        dict: a dictionary containing the prior for each cosmological parameter.
    """
    priors = {}
    for param in CONFIG.COSMO:
        priors[param] = generate_prior(CONFIG.PRIORS[param])
    return priors


class EmuPredict:
    """
    Predicts the linear matter power spectrum using the GP emulator.

    Args:
        nlhs (int, optional): The number of Latin Hypercube samples used. Defaults to 500.
    """

    def __init__(self, nlhs: int = 500, download: bool = True):
        self.config = CONFIG
        self.priors = generate_cosmo_prior()
        self.gps = load_gps(nlhs, CONFIG.MODEL_LINK, download)
        self.wavenumber = torch.logspace(
            np.log10(self.config.K_MIN), np.log10(self.config.K_MAX), self.config.NWAVE
        )

    def check_prior(self, redshift: float, parameter: dict) -> None:
        """
        Check the prior of the parameters (and redshift) and print a message if either is outside the box.

        Args:
            redshift (float): the redshift.
            parameter (dict): the cosmological parameter.
        """
        logprior = 0.0
        for name in ["Omega_cdm", "Omega_b", "h"]:
            logprior += self.priors[name].logpdf(parameter[name])
        logprior += self.priors["z"].logpdf(redshift)
        if not np.isfinite(logprior):
            print("Parameter is outside prior box. Unreliable predictions expected.")

    def calculate_prefactor(self, parameter: torch.tensor) -> torch.tensor:
        """
        Calculate the prefactor in terms of sigma8 and n_s.

        Args:
            parameter (torch.tensor): the two parameters: sigma8 and n_s.

        Returns:
            torch.tensor: the prefactor, the term containing sigma8 and n_s.
        """
        pre_sigma8 = (parameter[0] / self.config.FIX_SIGMA8) ** 2
        pre_ns = (self.wavenumber / self.config.FIX_K_PIVOT) ** (
            parameter[1] - self.config.FIX_NS
        )
        prefactor = pre_sigma8 * pre_ns
        return prefactor

    def calculate_gp_mean(self, redshift: float, parameter: dict) -> torch.tensor:
        """
        Calculate the mean of the GP. Note that the GP is built on top of:
        1) redshift
        2) Omega_cdm
        3) Omega_b
        4) h

        Args:
            redshift (float): the redshift value
            parameter (dict): the cosmological parameters

        Returns:
            torch.tensor: the predicted mean of the GP.
        """
        self.check_prior(redshift, parameter)
        param = torch.tensor(
            [parameter["Omega_cdm"], parameter["Omega_b"], parameter["h"], redshift]
        )
        pred = torch.tensor(
            [
                self.gps[i].predict_mean(param).data[0].item()
                for i in range(self.config.NWAVE)
            ]
        )
        return pred

    def calculate_gp_mean_var(
        self, redshift: float, parameter: dict
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Calculate BOTH the mean and the variance using the GP. Note that the GP is built on top of:
        1) redshift
        2) Omega_cdm
        3) Omega_b
        4) h

        Args:
            redshift (float): the redshift value
            parameter (dict): the cosmological parameters

        Returns:
            Tuple[torch.tensor, torch.tensor]: the GP mean and variance of the power spectrum
        """
        self.check_prior(redshift, parameter)
        param = torch.tensor(
            [parameter["Omega_cdm"], parameter["Omega_b"], parameter["h"], redshift]
        )
        preds = [self.gps[i].predict_mean_var(param) for i in range(self.config.NWAVE)]
        preds = list(map(torch.stack, zip(*preds)))
        gp_mean, gp_var = preds[0].view(-1), preds[1].view(-1)
        return gp_mean, gp_var

    def calculate_pklin(
        self, redshift: np.ndarray, cosmo: dict, return_var: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculates the final linear matter power spectrum.

        Args:
            redshift (float): the redshift at which we want to evaluate the linear power spectrum.
            cosmo (dict): a dictionary of the cosmological parameters.
            return_var (bool, optional): return the variance prediction on the linear Pk. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: mean or (mean and variance) of the linear Pk.
        """
        gp_cosmo = {
            "Omega_cdm": cosmo["Omega_cdm"],
            "Omega_b": cosmo["Omega_b"],
            "h": cosmo["h"],
        }

        # extra parameters
        pre_param = torch.tensor([cosmo["sigma8"], cosmo["n_s"]])
        prefactor = self.calculate_prefactor(pre_param)

        if return_var:
            gp_mean, gp_var = self.calculate_gp_mean_var(redshift, gp_cosmo)
            pred_pklin_mean = prefactor * gp_mean
            pred_pklin_var = prefactor**2 * gp_var
            pred_pklin_mean = pred_pklin_mean.numpy()
            pred_pklin_var = pred_pklin_var.numpy()
            pred_pklin_var[pred_pklin_var < 0.0] = 1e-32
            return pred_pklin_mean, pred_pklin_var

        gp_mean = self.calculate_gp_mean(redshift, gp_cosmo)
        pred_pklin = prefactor * gp_mean
        pred_pklin = pred_pklin.numpy()
        return pred_pklin
