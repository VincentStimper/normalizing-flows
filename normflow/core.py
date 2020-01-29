import torch
import torch.nn as nn
from . import distributions

class NormalizingFlow(nn.Module):
    """
    Normalizing flow model
    """
    def __init__(self, prior, decoder=None, q0=distributions.Dirac, flows=None):
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
        self.flows = flows
        self.q0 = q0

    def forward(self, x):
        z, log_q = self.q0(x)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.prior(z)
        if self.decoder is not None:
            log_p += self.decoder(x, z)
        return z, log_q, log_p