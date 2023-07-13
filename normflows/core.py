import torch
import torch.nn as nn
import numpy as np

from . import distributions
from . import utils


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, q0, flows, p=None):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
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
        """Alpha divergence when sampling from q

        Args:
          num_samples: Number of samples to draw
          dreg: Flag whether to use Double Reparametrized Gradient estimator, see [arXiv 1810.04152](https://arxiv.org/abs/1810.04152)

        Returns:
          Alpha divergence
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
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class ConditionalNormalizingFlow(NormalizingFlow):
    """
    Conditional normalizing flow model, providing condition,
    which is also called context, to both the base distribution
    and the flow layers
    """
    def forward(self, z, context=None):
        """Transforms latent variable z to the flow variable x

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z, context=context)
        return z

    def forward_and_log_det(self, z, context=None):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space
          context: Batch of conditions/context

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z, context=context)
            log_det += log_d
        return z, log_det

    def inverse(self, x, context=None):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x, context=context)
        return x

    def inverse_and_log_det(self, x, context=None):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution
          context: Batch of conditions/context

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x, context=context)
            log_det += log_d
        return x, log_det

    def sample(self, num_samples=1, context=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          context: Batch of conditions/context

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, context=context)
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, context=None):
        """Get log probability for batch

        Args:
          x: Batch
          context: Batch of conditions/context

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return log_q

    def forward_kld(self, x, context=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          context: Batch of conditions/context

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        log_q += self.q0.log_prob(z, context=context)
        return -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, context=None, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          num_samples: Number of samples to draw from base distribution
          context: Batch of conditions/context
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)

        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples, context=context)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z, context=context)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_, context=context)
                log_q += log_det
            log_q += self.q0.log_prob(z_, context=context)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z, context=context)
        return torch.mean(log_q) - beta * torch.mean(log_p)


class ClassCondFlow(nn.Module):
    """
    Class conditional normalizing Flow model, providing the
    class to be conditioned on only to the base distribution,
    as done e.g. in [Glow](https://arxiv.org/abs/1807.03039)
    """

    def __init__(self, q0, flows):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)

    def forward_kld(self, x, y):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return -torch.mean(log_q)

    def sample(self, num_samples=1, y=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, y)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, y):
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, y)
        return log_q

    def save(self, path):
        """Save state dict of model

        Args:
         param path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))


class MultiscaleFlow(nn.Module):
    """
    Normalizing Flow model with multiscale architecture, see RealNVP or Glow paper
    """

    def __init__(self, q0, flows, merges, transform=None, class_cond=True):
        """Constructor

        Args:

          q0: List of base distribution
          flows: List of list of flows for each level
          merges: List of merge/split operations (forward pass must do merge)
          transform: Initial transformation of inputs
          class_cond: Flag, indicated whether model has class conditional
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
        """Estimates forward KL divergence, see see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
          x: Batch sampled from target distribution
          y: Batch of targets, if applicable

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        return -torch.mean(self.log_prob(x, y))

    def forward(self, x, y=None):
        """Get negative log-likelihood for maximum likelihood training

        Args:
          x: Batch of data
          y: Batch of targets, if applicable

        Returns:
            Negative log-likelihood of the batch
        """
        return -self.log_prob(x, y)

    def forward_and_log_det(self, z):
        """Get observed variable x from list of latent variables z

        Args:
            z: List of latent variables

        Returns:
            Observed variable x, log determinant of Jacobian
        """
        log_det = torch.zeros(len(z[0]), dtype=z[0].dtype, device=z[0].device)
        for i in range(len(self.q0)):
            if i == 0:
                z_ = z[0]
            else:
                z_, log_det_ = self.merges[i - 1]([z_, z[i]])
                log_det += log_det_
            for flow in self.flows[i]:
                z_, log_det_ = flow(z_)
                log_det += log_det_
        if self.transform is not None:
            z_, log_det_ = self.transform(z_)
            log_det += log_det_
        return z_, log_det

    def inverse_and_log_det(self, x):
        """Get latent variable z from observed variable x

        Args:
            x: Observed variable

        Returns:
            List of latent variables z, log determinant of Jacobian
        """
        log_det = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        if self.transform is not None:
            x, log_det_ = self.transform.inverse(x)
            log_det += log_det_
        z = [None] * len(self.q0)
        for i in range(len(self.q0) - 1, -1, -1):
            for flow in reversed(self.flows[i]):
                x, log_det_ = flow.inverse(x)
                log_det += log_det_
            if i == 0:
                z[i] = x
            else:
                [x, z[i]], log_det_ = self.merges[i - 1].inverse(x)
                log_det += log_det_
        return z, log_det

    def sample(self, num_samples=1, y=None, temperature=None):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw
          y: Classes to sample from, will be sampled uniformly if None
          temperature: Temperature parameter for temp annealed sampling

        Returns:
          Samples, log probability
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
        """Get log probability for batch

        Args:
          x: Batch
          y: Classes of x

        Returns:
          log probability
        """
        log_q = 0
        z = x
        if self.transform is not None:
            z, log_det = self.transform.inverse(z)
            log_q += log_det
        for i in range(len(self.q0) - 1, -1, -1):
            for j in range(len(self.flows[i]) - 1, -1, -1):
                z, log_det = self.flows[i][j].inverse(z)
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
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))

    def set_temperature(self, temperature):
        """Set temperature for temperature a annealed sampling

        Args:
          temperature: Temperature parameter
        """
        for q0 in self.q0:
            if hasattr(q0, "temperature"):
                q0.temperature = temperature
            else:
                raise NotImplementedError(
                    "One base function does not "
                    "support temperature annealed sampling"
                )

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
        """Constructor of normalizing flow model

        Args:
          prior: Prior distribution of te VAE, i.e. Gaussian
          decoder: Optional decoder
          flows: Flows to transform output of base encoder
          q0: Base Encoder
        """
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.flows = nn.ModuleList(flows)
        self.q0 = q0

    def forward(self, x, num_samples=1):
        """Takes data batch, samples num_samples for each data point from base distribution

        Args:
          x: data batch
          num_samples: number of samples to draw for each data point

        Returns:
          latent variables for each batch and sample, log_q, and log_p
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
