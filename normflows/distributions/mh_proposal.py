import numpy as np
import torch
from torch import nn


class MHProposal(nn.Module):
    """
    Proposal distribution for the Metropolis Hastings algorithm
    """

    def __init__(self):
        super().__init__()

    def sample(self, z):
        """
        Sample new value based on previous z
        """
        raise NotImplementedError

    def log_prob(self, z_, z):
        """
        Args:
          z_: Potential new sample
          z: Previous sample

        Returns:
          Log probability of proposal distribution
        """
        raise NotImplementedError

    def forward(self, z):
        """Draw samples given z and compute log probability difference

        ```
        log(p(z | z_new)) - log(p(z_new | z))
        ```

        Args:
          z: Previous samples

        Returns:
          Proposal, difference of log probability ratio
        """
        raise NotImplementedError


class DiagGaussianProposal(MHProposal):
    """
    Diagonal Gaussian distribution with previous value as mean
    as a proposal for Metropolis Hastings algorithm
    """

    def __init__(self, shape, scale):
        """Constructor

        Args:
          shape: Shape of variables to sample
          scale: Standard deviation of distribution
        """
        super().__init__()
        self.shape = shape
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer("scale", self.scale_cpu.unsqueeze(0))

    def sample(self, z):
        num_samples = len(z)
        eps = torch.randn((num_samples,) + self.shape, dtype=z.dtype, device=z.device)
        z_ = eps * self.scale + z
        return z_

    def log_prob(self, z_, z):
        log_p = -0.5 * np.prod(self.shape) * np.log(2 * np.pi) - torch.sum(
            torch.log(self.scale) + 0.5 * torch.pow((z_ - z) / self.scale, 2),
            list(range(1, z.dim())),
        )
        return log_p

    def forward(self, z):
        num_samples = len(z)
        eps = torch.randn((num_samples,) + self.shape, dtype=z.dtype, device=z.device)
        z_ = eps * self.scale + z
        log_p_diff = torch.zeros(num_samples, dtype=z.dtype, device=z.device)
        return z_, log_p_diff
