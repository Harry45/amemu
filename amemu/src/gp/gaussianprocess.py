"""
Author: Dr. Arrykrishna Mootoovaloo
Date: 17th January 2022
Email: arrykrish@gmail.com, a.mootoovaloo17@imperial.ac.uk, arrykrishna.mootoovaloo@physics
Description: The zero mean Gaussian Process (noise-free implementation, otherwise an extension
of this is to also supply a noise covariance matrix)
"""

from typing import Tuple
import torch
import torch.autograd
import numpy as np
import src.gp.kernel as kn
import src.gp.transformation as tr


class GaussianProcess(tr.PreWhiten):
    """Zero mean Gaussian Process

    Args:
        inputs (torch.tensor): the inputs.
        outputs (torch.tensor): the outputs.
        jitter (float): the jitter term.
        xtrans (bool): whether to transform the inputs.
        ytrans (bool): whether to transform the outputs.
    """

    def __init__(self, inputs: torch.tensor, outputs: torch.tensor, jitter: float):
        # store the relevant informations
        self.jitter = jitter

        # get the dimensions of the inputs
        self.ndata, self.ndim = inputs.shape
        if self.ndim >= 2:
            tr.PreWhiten.__init__(self, inputs)
        assert self.ndata > self.ndim, "N < d, please reshape the inputs such that N > d."
        self.xtrain, self.ytrain, self.ymean, self.ystd = self._postinit(inputs, outputs)

        # store important quantities
        self.d_opt = None
        self.opt_parameters = None
        self.kernel_matrix = None
        self.alpha = None

    def _postinit(
        self, inputs: torch.tensor, outputs: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Initialise the training points.

        Args:
            inputs (torch.tensor): the inputs to the GP.
            outputs (torch.tensor): the target of the GP.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: xtrain, ytrain, ymean and ystd
        """

        # transform the inputs
        if self.ndim >= 2:
            xtrain = tr.PreWhiten.x_transformation(self, inputs)
        else:
            xtrain = inputs

        ylog = torch.log(outputs)
        ymean = torch.mean(ylog)
        ystd = torch.std(ylog)
        ytrain = (ylog - ymean) / ystd
        ytrain = ytrain.view(-1, 1)
        return xtrain, ytrain, ymean, ystd

    def cost(self, parameters: torch.tensor) -> torch.tensor:
        """Calculates the negative log-likelihood of the GP, for fitting the kernel hyperparameters.

        Args:
            parameters (torch.tensor): the set of input parameters.

        Returns:
            torch.tensor: the value of the negative log-likelihood.
        """

        # compute the kernel matrix
        kernel = kn.compute(self.xtrain, self.xtrain, parameters)

        # add the jitter term to the kernel matrix
        kernel = kernel + torch.eye(self.xtrain.shape[0]) * self.jitter

        # compute the chi2 and log-determinant of the kernel matrix
        log_marginal = -0.5 * self.ytrain.t() @ kn.solve(kernel, self.ytrain) - 0.5 * kn.logdeterminant(kernel)
        return -log_marginal

    def optimisation(
        self,
        parameters: torch.tensor,
        niter: int = 10,
        lrate: float = 0.01,
        nrestart: int = 5,
    ) -> dict:
        """Optimise for the kernel hyperparameters using Adam in PyTorch.

        Args:
            parameters(torch.tensor): a tensor of the kernel hyperparameters.
            niter(int): the number of iterations we want to use
            lr(float): the learning rate
            nrestart(int): the number of times we want to restart the optimisation

        Returns:
            dict: dictionary consisting of the optimised values of the hyperparameters and the loss.
        """

        dictionary = {}

        for i in range(nrestart):
            # make a copy of the original parameters and perturb it
            params = torch.randn(parameters.shape)  # parameters.clone() + torch.randn(parameters.shape) * 0.1

            # make sure we are differentiating with respect to the parameters
            params.requires_grad = True

            # initialise the optimiser
            optimiser = torch.optim.Adam([params], lr=lrate)

            loss = self.cost(params)

            # an empty list to store the loss
            record_loss = [loss.item()]

            # run the optimisation
            for _ in range(niter):
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # evaluate the loss
                loss = self.cost(params)

                # record the loss at every step
                record_loss.append(loss.item())

            dictionary[i] = {"parameters": params, "loss": record_loss}

        # get the dictionary for which the loss is the lowest
        self.d_opt = dictionary[np.argmin([dictionary[i]["loss"][-1] for i in range(nrestart)])]

        # store the optimised parameters as well
        self.opt_parameters = self.d_opt["parameters"]

        # compute the kernel and store it
        self.kernel_matrix = kn.compute(self.xtrain, self.xtrain, self.opt_parameters.data)

        # also compute K^-1 y and store it
        self.alpha = kn.solve(self.kernel_matrix, self.ytrain)

        # return the optimised values of the hyperparameters and the loss
        return dictionary

    def predict_mean(self, testpoint: torch.tensor) -> torch.tensor:
        """Calculates the mean prediction of the GP.

        Args:
            testpoint(torch.tensor): the test point.

        Returns:
            torch.tensor: the mean prediction from the GP
        """
        testpoint = testpoint.view(-1, 1)
        if self.ndim >= 2:
            testpoint = tr.PreWhiten.x_transformation(self, testpoint)

        k_star = kn.compute(self.xtrain, testpoint, self.opt_parameters.data)
        mean = k_star.t() @ self.alpha
        mean = torch.exp(mean * self.ystd + self.ymean)
        return mean.view(-1)

    def gradient(self, testpoint: torch.tensor) -> torch.tensor:
        """Calculates the gradient of the GP.

        Args:
            testpoint(torch.tensor): the test point.

        Returns:
            torch.tensor: the gradient of the GP with respect to the inputs
        """

        testpoint.requires_grad = True
        mean = self.predict_mean(testpoint)
        grad = torch.autograd.grad(mean, testpoint)[0]
        testpoint.requires_grad = False
        return grad

    def hessian(self, testpoint: torch.tensor) -> torch.tensor:
        """
        Calculates the second derivatives of the GP with respect to the test point.

        Args:
            testpoint (torch.tensor): the input test point.

        Returns:
            torch.tensor: the second derivatives
        """
        testpoint.requires_grad = True
        hess = torch.autograd.functional.hessian(self.predict_mean, testpoint)
        testpoint.requires_grad = False
        return hess

    def predict_mean_var(self, testpoint: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """Computes the prediction at a given test point.

        Args:
            testpoint(torch.tensor): a tensor of the test point
            variance(bool, optional): if we want to compute the variance as well. Defaults to False.

        Returns:
            Tuple[torch.tensor, torch.tensor]: The mean and variance
        """
        testpoint = testpoint.view(-1, 1)
        if self.ndim >= 2:
            testpoint = tr.PreWhiten.x_transformation(self, testpoint)
        k_star = kn.compute(self.xtrain, testpoint, self.opt_parameters.data)
        k_star_star = kn.compute(testpoint, testpoint, self.opt_parameters.data)

        # the mean prediction
        mean = k_star.t() @ self.alpha
        mean = torch.exp(mean * self.ystd + self.ymean)

        # the variance calculation
        var = k_star_star - k_star.t() @ kn.solve(self.kernel_matrix, k_star)
        var = (self.ystd * mean) ** 2 * var
        return mean.view(-1), var.view(-1)

    def del_kernel(self):
        """
        Delete the kernel matrix for reducing storage size of GP.
        """
        del self.kernel_matrix
