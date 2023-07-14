# Author: Arrykrishna Mootoovaloo
# Date: January 2022
# Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
# Description: In this code, we train the GP model using the training data.

import os
from pathlib import Path
import torch
import matplotlib.pylab as plt

# our script and functions
from amemu.src.gp.gaussianprocess import GaussianProcess
from amemu.utils.helpers import load_csv, save_list
import amemu.config as CONFIG

plt.rc("text", usetex=True)
plt.rc("font", **{"family": "sans-serif", "serif": ["Palatino"]})
FONTSIZE = 20


def plot_loss(optim: dict, fname: str, save: bool = True):
    """Plots the loss of the GP model.

    Args:
        optim (dict): A dictionary with the optimizer. There can be more than 1 optimisation.
        save (bool, optional): Whether to save the plot. Defaults to True.
    """

    nopt = len(optim)
    niter = len(optim[0]["loss"])

    # plot the loss
    plt.figure(figsize=(8, 8))
    for i in range(nopt):
        plt.plot(range(niter), optim[i]["loss"], label=f"Optimization {i + 1}")
    plt.xlabel("Iterations", fontsize=FONTSIZE)
    plt.ylabel("Loss", fontsize=FONTSIZE)
    plt.tick_params(axis="x", labelsize=FONTSIZE)
    plt.tick_params(axis="y", labelsize=FONTSIZE)
    plt.legend(loc="best", prop={"family": "sans-serif", "size": 15})
    plt.savefig(fname + ".pdf", bbox_inches="tight")
    plt.savefig(fname + ".png", bbox_inches="tight")
    if save:
        plt.close()
    else:
        plt.show()


def train_gps(nlhs: int, jitter: float = 1e-6) -> list:
    """Train the Gaussian Processes and store them.

    Args:
        nlhs (int): number of Latin Hypercube samples.
        xtrans (bool): the transformation for the inputs.
        ytrans (bool): the transformation for the outputs.
        jitter (float, optional): the jitter term for numerical stability. Defaults to 1E-6.

    Returns:
        list: A list of Gaussian Processes.
    """
    # paths for the data, GP and loss
    parent_path = Path(__file__).parents[2]
    data_path = os.path.join(parent_path, "data")
    gp_path = os.path.join(parent_path, f"gps/{nlhs}")
    plot_path = os.path.join(parent_path, "results/loss")
    os.makedirs(gp_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    inputs = load_csv(data_path, "cosmologies_lhs_" + str(nlhs))
    outputs = load_csv(data_path, "pk_linear_lhs_" + str(nlhs))
    ins = torch.from_numpy(inputs.values)

    gps = []

    for i in range(CONFIG.NWAVE):
        print(f"Training GP {i + 1}")

        out = torch.from_numpy(outputs.iloc[:, i].values)

        # the GP module
        gp_module = GaussianProcess(ins, out, jitter)

        # perform the optimisation of the GP model
        opt = gp_module.optimisation(
            torch.randn(inputs.shape[1] + 1),
            niter=CONFIG.NITER,
            lrate=CONFIG.LEARN_RATE,
            nrestart=CONFIG.NRESTART,
        )

        gps.append(gp_module)

        # plot and store the loss function of the GP model
        plot_loss(opt, f"{plot_path}/pk_linear_lhs_" + str(nlhs) + "_wave_" + str(i))

        # name of the files to save
        gp_name = "pk_linear_lhs_" + str(nlhs) + "_wave_" + str(i)
        pa_name = "params_" + str(nlhs) + "_wave_" + str(i)
        al_name = "alpha_" + str(nlhs) + "_wave_" + str(i)

        save_list(gp_module, gp_path, gp_name)
        save_list(gp_module.opt_parameters.data.numpy(), gp_path, pa_name)
        save_list(gp_module.alpha.data.numpy(), gp_path, al_name)
    return gps
