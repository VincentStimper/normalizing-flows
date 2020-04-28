import torch
import torch.nn as nn
from . import distributions
from . import utils

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
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1., score_fn=True):
        """
        Estimates reverse KL divergence, see arXiv 1912.02762
        :param num_samples: Number of samples to draw from base distribution
        :param beta: Annealing parameter, see arXiv 1505.05770
        :param score_fn: Flag whether to include score function in gradient, see
        arXiv 1703.09194
        :return: Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        """
        Alpha divergence when sampling from q
        :param num_samples: Number of samples to draw
        :param dreg: Flag whether to use Double REparametrized Gradient estimator,
        see arXiv 1810.04152
        :return:Alpha divergence
        """
        if dreg:
            pass
        else:
            z, log_q = self.q0(num_samples)
            for flow in self.flows:
                z, log_det = flow(z)
                log_q -= log_det
            log_p = self.p.log_prob(z)
            w = torch.exp(alpha * (log_p - log_q))
            loss = -torch.mean(torch.log(torch.mean(w, 1)))
        return loss

    def sample(self, num_samples=1):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :return: Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x):
        """
        Get log probability for batch
        :param x: Batch
        :return: log probability
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """
        Save state dict of model
        :param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load model from state dict
        :param path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


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