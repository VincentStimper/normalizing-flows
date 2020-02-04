import torch
import torch.nn as nn
import numpy as np

class ParametrizedConditionalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_sample=1):
        """
        :param x: Variable to condition on, first dimension is batch size
        :param num_sample: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        raise NotImplementedError


class Dirac(ParametrizedConditionalDistribution):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_sample=1):
        z = x.unsqueeze(1).repeat(1, num_sample, 1)
        log_p = torch.zeros(z.size()[0:2])
        return z, log_p


class ConstDiagGaussian(ParametrizedConditionalDistribution):
    def __init__(self, loc, scale):
        """
        Multivariate Gaussian distribution with diagonal covariance and parameters being constant wrt x
        :param loc: mean vector of the distribution
        :param scale: vector of the standard deviations on the diagonal of the covariance matrix
        """
        super().__init__()
        self.n = len(loc)
        if not torch.is_tensor(loc):
            loc = torch.tensor(loc)
        if not torch.is_tensor(scale):
            scale = torch.is_tensor(scale)
        self.loc = nn.Parameter(loc.reshape((1, 1, self.n)))
        self.scale = nn.Parameter(scale.reshape((1, 1, self.n)))
        self.eps_dist = torch.distributions.normal.Normal(0, 1)


    def forward(self, x=None, num_sample=1):
        """
        :param x: Variable to condition on, will only be used to determine the batch size
        :param num_sample: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        if x is not None:
            batch_size = x.size()
        else:
            batch_size = 1
        eps = self.eps_dist.sample((batch_size, num_sample, self.n))
        z = self.loc + self.scale * eps
        log_p = - 0.5 * self.n * np.log(2 * np.pi) - 0.5 * torch.sum(torch.log(self.scale) + ((z - self.loc) / self.scale) ** 2, 2)
        return z, log_p


class PriorDistribution:
    def __init__(self):
        raise NotImplementedError

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        raise NotImplementedError


class TwoModes(PriorDistribution):
    def __init__(self, loc, scale):
        """
        Distribution 2d with two modes at z[0] = -loc and z[0] = loc
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.loc = loc
        self.scale = scale

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        assert z.dim() == 2
        log_prob = - 0.5 * ((torch.norm(z, dim=1) - self.loc) / (2 * self.scale)) ** 2\
                   + torch.log(torch.exp(-0.5 * ((z[:, 0] - self.loc) / (3 * self.scale)) ** 2)
                               + torch.exp(-0.5 * ((z[:, 0] + self.loc) / (3 * self.scale)) ** 2))
        return log_prob