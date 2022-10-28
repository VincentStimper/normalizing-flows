import numpy as np
import torch
from torch import nn


class Target(nn.Module):
    """
    Sample target distributions to test models
    """

    def __init__(self, prop_scale=torch.tensor(6.0), prop_shift=torch.tensor(-3.0)):
        """Constructor

        Args:
          prop_scale: Scale for the uniform proposal
          prop_shift: Shift for the uniform proposal
        """
        super().__init__()
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        raise NotImplementedError("The log probability is not implemented yet.")

    def rejection_sampling(self, num_steps=1):
        """Perform rejection sampling on image distribution

        Args:
          num_steps: Number of rejection sampling steps to perform

        Returns:
          Accepted samples
        """
        eps = torch.rand(
            (num_steps, self.n_dims),
            dtype=self.prop_scale.dtype,
            device=self.prop_scale.device,
        )
        z_ = self.prop_scale * eps + self.prop_shift
        prob = torch.rand(
            num_steps, dtype=self.prop_scale.dtype, device=self.prop_scale.device
        )
        prob_ = torch.exp(self.log_prob(z_) - self.max_log_prob)
        accept = prob_ > prob
        z = z_[accept, :]
        return z

    def sample(self, num_samples=1):
        """Sample from image distribution through rejection sampling

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples
        """
        z = torch.zeros(
            (0, self.n_dims), dtype=self.prop_scale.dtype, device=self.prop_scale.device
        )
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z


class TwoMoons(Target):
    """
    Bimodal two-dimensional distribution
    """

    def __init__(self):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0

    def log_prob(self, z):
        """
        ```
        log(p) = - 1/2 * ((norm(z) - 2) / 0.2) ** 2
                 + log(  exp(-1/2 * ((z[0] - 2) / 0.3) ** 2)
                       + exp(-1/2 * ((z[0] + 2) / 0.3) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )
        return log_prob


class CircularGaussianMixture(nn.Module):
    """
    Two-dimensional Gaussian mixture arranged in a circle
    """

    def __init__(self, n_modes=8):
        """Constructor

        Args:
          n_modes: Number of modes
        """
        super(CircularGaussianMixture, self).__init__()
        self.n_modes = n_modes
        self.register_buffer(
            "scale", torch.tensor(2 / 3 * np.sin(np.pi / self.n_modes)).float()
        )

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_modes):
            d_ = (
                (z[:, 0] - 2 * np.sin(2 * np.pi / self.n_modes * i)) ** 2
                + (z[:, 1] - 2 * np.cos(2 * np.pi / self.n_modes * i)) ** 2
            ) / (2 * self.scale**2)
            d = torch.cat((d, d_[:, None]), 1)
        log_p = -torch.log(
            2 * np.pi * self.scale**2 * self.n_modes
        ) + torch.logsumexp(-d, 1)
        return log_p

    def sample(self, num_samples=1):
        eps = torch.randn(
            (num_samples, 2), dtype=self.scale.dtype, device=self.scale.device
        )
        phi = (
            2
            * np.pi
            / self.n_modes
            * torch.randint(0, self.n_modes, (num_samples,), device=self.scale.device)
        )
        loc = torch.stack((2 * torch.sin(phi), 2 * torch.cos(phi)), 1).type(eps.dtype)
        return eps * self.scale + loc


class RingMixture(Target):
    """
    Mixture of ring distributions in two dimensions
    """

    def __init__(self, n_rings=2):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.0
        self.n_rings = n_rings
        self.scale = 1 / 4 / self.n_rings

    def log_prob(self, z):
        d = torch.zeros((len(z), 0), dtype=z.dtype, device=z.device)
        for i in range(self.n_rings):
            d_ = ((torch.norm(z, dim=1) - 2 / self.n_rings * (i + 1)) ** 2) / (
                2 * self.scale**2
            )
            d = torch.cat((d, d_[:, None]), 1)
        return torch.logsumexp(-d, 1)
