### Implementation of Hamiltonian Annealed Importance Sampling ###

import torch
from .. import distributions
from .. import flows


class HAIS:
    """
    Class which performs HAIS
    """

    def __init__(self, betas, prior, target, num_leapfrog, step_size, log_mass):
        """
        Args:
          betas: Annealing schedule, the jth target is ```f_j(x) = f_0(x)^{\beta_j} f_n(x)^{1-\beta_j}``` where the target is proportional to f_0 and the prior is proportional to f_n. The number of intermediate steps is infered from the shape of betas. Should be of the form 1 = \beta_0 > \beta_1 > ... > \beta_n = 0
          prior: The prior distribution to start the HAIS chain.
          target: The target distribution from which we would like to draw weighted samples.
          num_leapfrog: Number of leapfrog steps in the HMC transitions.
          step_size: step_size to use for HMC transitions.
          log_mass: log_mass to use for HMC transitions.
        """
        self.prior = prior
        self.target = target
        self.layers = []
        n = betas.shape[0] - 1
        for i in range(n - 1, 0, -1):
            intermediate_target = distributions.LinearInterpolation(
                self.target, self.prior, betas[i]
            )
            self.layers += [
                flows.HamiltonianMonteCarlo(
                    intermediate_target, num_leapfrog, torch.log(step_size), log_mass
                )
            ]

    def sample(self, num_samples):
        """Run HAIS to draw samples from the target with appropriate weights.

        Args:
          num_samples: The number of samples to draw.a
        """
        samples, log_weights = self.prior.forward(num_samples)
        log_weights = -log_weights
        for i in range(len(self.layers)):
            samples, log_weights_addition = self.layers[i].forward(samples)
            log_weights += log_weights_addition
        log_weights += self.target.log_prob(samples)
        return samples, log_weights
