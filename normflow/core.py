import torch
import torch.nn as nn
from . import distributions

class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """
    def __init__(self, q0, flows, p=None):
        """
        Constructor
        :param q0: Base distribution
        :param flows: List of flows
        :param p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward_kld(self, x):
        """
        Estimates forward KL divergence, see arXiv 1912.02762
        :param x: Batch sampled from target distribution
        :return: Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x))
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples):
        """
        Estimates reverse KL divergence, see arXiv 1912.02762
        :param num_samples: Number of samples to draw from base distribution
        :return: Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q = self.q0.sample(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - torch.mean(log_p)


class NormalizingFlowVAE(nn.Module):
    """
    VAE using normalizing flows to express approximate distribution
    """
    def __init__(self, prior, q0=distributions.Dirac(), flows=None, decoder=None):
        """
        Constructor of normalizing flow model
        :param prior: Prior distribution of te VAE, i.e. Gaussian
        :param decoder: Optional decoder
        :param flows: Flows to transform output of base encoder
        :param q0: Base Encoder
        """
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.flows = nn.ModuleList(flows)
        self.q0 = q0

    def forward(self, x, num_samples=1):
        """
        Takes data batch, samples num_samples for each data point from base distribution
        :param x: data batch
        :param num_samples: number of samples to draw for each data point
        :return: latent variables for each batch and sample, log_q, and log_p
        """
        z, log_q = self.q0(x, num_samples=num_samples)
        # Flatten batch and sample dim
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.prior.log_prob(z)
        if self.decoder is not None:
            log_p += self.decoder.log_prob(x, z)
        # Separate batch and sample dimension again
        z = z.view(-1, num_samples, *z.size()[1:])
        log_q = log_q.view(-1, num_samples, *log_q.size()[1:])
        log_p = log_p.view(-1, num_samples, *log_p.size()[1:])
        return z, log_q, log_p