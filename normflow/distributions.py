import torch
import torch.nn as nn

class ParametrizedConditionalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: Variable to condition on, first dimension is batch size
        :return: sample of z for x, log probability for sample
        """
        raise NotImplementedError


class Dirac(ParametrizedConditionalDistribution):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = x
        if x.size() == torch.Size([]):
            log_p = torch.zeros(x.size())
        else:
            log_p = torch.zeros(x.size()[0])
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