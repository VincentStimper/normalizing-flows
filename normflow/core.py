import torch
import torch.nn as nn
from . import distributions
from . import utils
import math

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
        :param dreg: Flag whether to use Double Reparametrized Gradient estimator,
        see arXiv 1810.04152
        :return: Alpha divergence
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const ** alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha ** 2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = -torch.logsumexp(alpha * (log_p - log_q), 0) + \
                math.log(log_p.shape[0])
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
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
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


class ClassCondFlow(nn.Module):
    """
    Class conditional normalizing Flow model
    """
    def __init__(self, q0, flows):
        """
        Constructor
        :param q0: Base distribution
        :param flows: List of flows
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)

    def forward_kld(self, x, y):
        """
        Estimates forward KL divergence, see arXiv 1912.02762
        :param x: Batch sampled from target distribution
        :return: Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return -torch.mean(log_q)

    def sample(self, num_samples=1, y=None):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :param y: Classes to sample from, will be sampled uniformly if None
        :return: Samples, log probability
        """
        z, log_q = self.q0(num_samples, y)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, y):
        """
        Get log probability for batch
        :param x: Batch
        :param y: Classes of x
        :return: log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
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


class MultiscaleFlow(nn.Module):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """
    def __init__(self, q0, flows, merges, transform=None, class_cond=True):
        """
        Constructor
        :param q0: List of base distribution
        :param flows: List of list of flows for each level
        :param merges: List of merge/split operations (forward pass must do merge)
        :param transform: Initial transformation of inputs
        :param class_cond: Flag, indicated whether model has class conditional
        base distributions
        """
        super().__init__()
        self.q0 = nn.ModuleList(q0)
        self.num_levels = len(self.q0)
        self.flows = torch.nn.ModuleList([nn.ModuleList(flow) for flow in flows])
        self.merges = torch.nn.ModuleList(merges)
        self.transform = transform
        self.class_cond = class_cond

    def forward_kld(self, x, y=None):
        """
        Estimates forward KL divergence, see arXiv 1912.02762
        :param x: Batch sampled from target distribution
        :return: Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, y))

    def forward(self, x, y=None):
        """
        Get negative log-likelihood for maximum likelihood training
        :param x: Batch of data
        :param y: Batch of targets, if applicable
        :return: NLL
        """
        return -self.log_prob(x, y)

    def sample_prior(self, num_samples):
        samples = []
        for i in range(len(self.q0)):
            if self.class_cond:
                z_, log_q_ = self.q0[i](num_samples, y)
            else:
                z_, log_q_ = self.q0[i](num_samples)
            samples.append((z_, log_q_))
        return samples

    def generate(self, latents, y=None, temperature=None):
        if temperature is not None:
            self.set_temperature(temperature)
        for i in range(len(self.q0)):
            z_, log_q_ = latents[i]
            if i == 0:
                log_q = log_q_
                z = z_
            else:
                log_q += log_q_
                z, log_det = self.merges[i - 1]([z, z_])
                log_q -= log_det
            for flow in self.flows[i]:
                z, log_det = flow(z)
                log_q -= log_det
        if self.transform is not None:
            z, log_det = self.transform(z)
            log_q -= log_det
        if temperature is not None:
            self.reset_temperature()
        return z, log_q

    def sample(self, num_samples=1, y=None, temperature=None):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :param y: Classes to sample from, will be sampled uniformly if None
        :param temperature: Temperature parameter for temp annealed sampling
        :return: Samples, log probability
        """
        if temperature is not None:
            self.set_temperature(temperature)
        for i in range(len(self.q0)):
            if self.class_cond:
                z_, log_q_ = self.q0[i](num_samples, y)
            else:
                z_, log_q_ = self.q0[i](num_samples)
            if i == 0:
                log_q = log_q_
                z = z_
            else:
                log_q += log_q_
                z, log_det = self.merges[i - 1]([z, z_])
                log_q -= log_det
            for flow in self.flows[i]:
                z, log_det = flow(z)
                log_q -= log_det
        if self.transform is not None:
            z, log_det = self.transform(z)
            log_q -= log_det
        if temperature is not None:
            self.reset_temperature()
        return z, log_q

    def log_prob(self, x, y):
        """
        Get log probability for batch
        :param x: Batch
        :param y: Classes of x
        :return: log probability
        """
        log_q = 0
        z = x
        if self.transform is not None:
            z, log_det = self.transform.inverse(z)
            log_q += log_det
        for i in range(len(self.q0) - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
                # z, log_det = torch.utils.checkpoint.checkpoint(self.flows[i][j].inverse, z)
                log_q += log_det
            if i > 0:
                [z, z_], log_det = self.merges[i - 1].inverse(z)
                log_q += log_det
            else:
                z_ = z
            if self.class_cond:
                log_q += self.q0[i].log_prob(z_, y)
            else:
                log_q += self.q0[i].log_prob(z_)
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

    def set_temperature(self, temperature):
        """
        Set temperature for temperature a annealed sampling
        :param temperature: Temperature parameter
        """
        for q0 in self.q0:
            if hasattr(q0, 'temperature'):
                q0.temperature = temperature
            else:
                raise NotImplementedError('One base function does not '
                                          'support temperature annealed sampling')

    def reset_temperature(self):
        """
        Set temperature values of base distributions back to None
        """
        self.set_temperature(None)


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
