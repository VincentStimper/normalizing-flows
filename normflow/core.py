import torch
import torch.nn as nn
from . import distributions

class NormalizingFlow(nn.Module):
    """
    Normalizing flow model
    """
    def __init__(self, prior, decoder=None, q0=distributions.Dirac(), flows=None):
        """
        Constructor of normalizing flow model
        :param prior:
        :param decoder:
        :param flows:
        :param q0:
        """
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.flows = nn.ModuleList(flows)
        self.q0 = q0

    def forward(self, x, num_samples=1):
        z, log_q = self.q0(x, num_samples=num_samples)
        print(z.size())
        print(log_q.size())
        for flow in self.flows:
            z, log_det = flow(z)
            print(z.size())
            print(log_det.size())
            log_q -= log_det
        log_p = self.prior.log_prob(z)
        if self.decoder is not None:
            log_p += self.decoder.log_p(x, z)
        return z, log_q, log_p

    def log_q(self, z, x):
        """
        :param z: Latent variable, first dimension is batch dimension
        :param x: Observable, first dimension is batch dimension
        :return: Approximate posterior at z given x
        """
        log_q = self.q0.log_prob(z, x)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return log_q