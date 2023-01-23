import numpy as np
import torch
from torch import nn


class BaseDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """Decodes z to x

        Args:
          z: latent variable

        Returns:
          x, std of x
        """
        raise NotImplementedError

    def log_prob(self, x, z):
        """Log probability

        Args:
          x: observable
          z: latent variable

        Returns:
          log(p) of x given z
        """
        raise NotImplementedError


class NNDiagGaussianDecoder(BaseDecoder):
    """
    BaseDecoder representing a diagonal Gaussian distribution with mean and std parametrized by a NN
    """

    def __init__(self, net):
        """Constructor

        Args:
          net: neural network parametrizing mean and standard deviation of diagonal Gaussian
        """
        super().__init__()
        self.net = net

    def forward(self, z):
        mean_std = self.net(z)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...]
        std = torch.exp(0.5 * mean_std[:, n_hidden:, ...])
        return mean, std

    def log_prob(self, x, z):
        mean_std = self.net(z)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...]
        var = torch.exp(mean_std[:, n_hidden:, ...])
        if len(z) > len(x):
            x = x.unsqueeze(1)
            x = x.repeat(1, z.size()[0] // x.size()[0], *((x.dim() - 2) * [1])).view(
                -1, *x.size()[2:]
            )
        log_p = -0.5 * torch.prod(torch.tensor(z.size()[1:])) * np.log(
            2 * np.pi
        ) - 0.5 * torch.sum(
            torch.log(var) + (x - mean) ** 2 / var, list(range(1, z.dim()))
        )
        return log_p


class NNBernoulliDecoder(BaseDecoder):
    """
    BaseDecoder representing a Bernoulli distribution with mean parametrized by a NN
    """

    def __init__(self, net):
        """Constructor

        Args:
          net: neural network parametrizing mean Bernoulli (mean = sigmoid(nn_out)
        """
        super().__init__()
        self.net = net

    def forward(self, z):
        mean = torch.sigmoid(self.net(z))
        return mean

    def log_prob(self, x, z):
        score = self.net(z)
        if len(z) > len(x):
            x = x.unsqueeze(1)
            x = x.repeat(1, z.size()[0] // x.size()[0], *((x.dim() - 2) * [1])).view(
                -1, *x.size()[2:]
            )
        log_sig = lambda a: -torch.relu(-a) - torch.log(1 + torch.exp(-torch.abs(a)))
        log_p = torch.sum(
            x * log_sig(score) + (1 - x) * log_sig(-score), list(range(1, x.dim()))
        )
        return log_p
