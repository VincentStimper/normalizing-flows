import numpy as np
import torch
from torch import nn



class BaseEncoder(nn.Module):
    """
    Base distribution of a flow-based variational autoencoder
    Parameters of the distribution depend of the target variable x
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        """
        :param x: Variable to condition on, first dimension is batch size
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        raise NotImplementedError

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch size
        :param x: Variable to condition on, first dimension is batch size
        :return: log probability of z given x
        """
        raise NotImplementedError


class Dirac(BaseEncoder):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1)
        log_p = torch.zeros(z.size()[0:2])
        return z, log_p

    def log_prob(self, z, x):
        log_p = torch.zeros(z.size()[0:2])
        return log_p


class Uniform(BaseEncoder):
    def __init__(self, zmin=0.0, zmax=1.0):
        super().__init__()
        self.zmin = zmin
        self.zmax = zmax
        self.log_p = -torch.log(zmax - zmin)

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1).uniform_(min=self.zmin, max=self.zmax)
        log_p = torch.zeros(z.size()[0:2]).fill_(self.log_p)
        return z, log_p

    def log_prob(self, z, x):
        log_p = torch.zeros(z.size()[0:2]).fill_(self.log_p)
        return log_p


class ConstDiagGaussian(BaseEncoder):
    def __init__(self, loc, scale):
        """
        Multivariate Gaussian distribution with diagonal covariance and parameters being constant wrt x
        :param loc: mean vector of the distribution
        :param scale: vector of the standard deviations on the diagonal of the covariance matrix
        """
        super().__init__()
        self.d = len(loc)
        if not torch.is_tensor(loc):
            loc = torch.tensor(loc).float()
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale).float()
        self.loc = nn.Parameter(loc.reshape((1, 1, self.d)))
        self.scale = nn.Parameter(scale)

    def forward(self, x=None, num_samples=1):
        """
        :param x: Variable to condition on, will only be used to determine the batch size
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        if x is not None:
            batch_size = len(x)
        else:
            batch_size = 1
        eps = torch.randn((batch_size, num_samples, self.d), device=x.device)
        z = self.loc + self.scale * eps
        log_p = - 0.5 * self.d * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * torch.pow(eps, 2), 2)
        return z, log_p

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch dimension
        :param x: Variable to condition on, first dimension is batch dimension
        :return: log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        log_p = - 0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            torch.log(self.scale) + 0.5 * ((z - self.loc) / self.scale) ** 2, 2)
        return log_p


class NNDiagGaussian(BaseEncoder):
    """
    Diagonal Gaussian distribution with mean and variance determined by a neural network
    """

    def __init__(self, net):
        """
        Constructor
        :param net: net computing mean (first n / 2 outputs), standard deviation (second n / 2 outputs)
        """
        super().__init__()
        self.net = net

    def forward(self, x, num_samples=1):
        """
        :param x: Variable to condition on
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        batch_size = len(x)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        std = torch.exp(0.5 * mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1))
        eps = torch.randn((batch_size, num_samples) + tuple(mean.size()[2:]), device=x.device)
        z = mean + std * eps
        log_p = - 0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi) \
                - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(2, z.dim())))
        return z, log_p

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch dimension
        :param x: Variable to condition on, first dimension is batch dimension
        :return: log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        var = torch.exp(mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1))
        log_p = - 0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi) \
                - 0.5 * torch.sum(torch.log(var) + (z - mean) ** 2 / var, 2)
        return log_p