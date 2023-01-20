import torch

from .base import Flow


class MetropolisHastings(Flow):
    """Sampling through Metropolis Hastings in Stochastic Normalizing Flow

    See [arXiv: 2002.06707](https://arxiv.org/abs/2002.06707)
    """

    def __init__(self, target, proposal, steps):
        """Constructor

        Args:
          target: The stationary distribution of this Markov transition, i.e. the target distribution to sample from.
          proposal: Proposal distribution
          steps: Number of MCMC steps to perform
        """
        super().__init__()
        self.target = target
        self.proposal = proposal
        self.steps = steps

    def forward(self, z):
        # Initialize number of samples and log(det)
        num_samples = len(z)
        log_det = torch.zeros(num_samples, dtype=z.dtype, device=z.device)
        # Get log(p) for current samples
        log_p = self.target.log_prob(z)
        for i in range(self.steps):
            # Make proposal and get log(p)
            z_, log_p_diff = self.proposal(z)
            log_p_ = self.target.log_prob(z_)
            # Make acceptance decision
            w = torch.rand(num_samples, dtype=z.dtype, device=z.device)
            log_w_accept = log_p_ - log_p + log_p_diff
            w_accept = torch.clamp(torch.exp(log_w_accept), max=1)
            accept = w <= w_accept
            # Update samples, log(det), and log(p)
            z = torch.where(accept.unsqueeze(1), z_, z)
            log_det_ = log_p - log_p_
            log_det = torch.where(accept, log_det + log_det_, log_det)
            log_p = torch.where(accept, log_p_, log_p)
        return z, log_det

    def inverse(self, z):
        # Equivalent to forward pass
        return self.forward(z)


class HamiltonianMonteCarlo(Flow):
    """Flow layer using the HMC proposal in Stochastic Normalising Flows

    See [arXiv: 2002.06707](https://arxiv.org/abs/2002.06707)
    """

    def __init__(self, target, steps, log_step_size, log_mass, max_abs_grad=None):
        """Constructor

        Args:
          target: The stationary distribution of this Markov transition, i.e. the target distribution to sample from.
          steps: The number of leapfrog steps
          log_step_size: The log step size used in the leapfrog integrator. shape (dim)
          log_mass: The log_mass determining the variance of the momentum samples. shape (dim)
          max_abs_grad: Maximum absolute value of the gradient of the target distribution's log probability. If set to None then no gradient clipping is applied. Useful for improving numerical stability."""
        super().__init__()
        self.target = target
        self.steps = steps
        self.register_parameter("log_step_size", torch.nn.Parameter(log_step_size))
        self.register_parameter("log_mass", torch.nn.Parameter(log_mass))
        self.max_abs_grad = max_abs_grad

    def forward(self, z):
        # Draw momentum
        p = torch.randn_like(z) * torch.exp(0.5 * self.log_mass)

        # leapfrog
        z_new = z.clone()
        p_new = p.clone()
        step_size = torch.exp(self.log_step_size)
        for i in range(self.steps):
            p_half = p_new - (step_size / 2.0) * -self.gradlogP(z_new)
            z_new = z_new + step_size * (p_half / torch.exp(self.log_mass))
            p_new = p_half - (step_size / 2.0) * -self.gradlogP(z_new)

        # Metropolis Hastings correction
        probabilities = torch.exp(
            self.target.log_prob(z_new)
            - self.target.log_prob(z)
            - 0.5 * torch.sum(p_new**2 / torch.exp(self.log_mass), 1)
            + 0.5 * torch.sum(p**2 / torch.exp(self.log_mass), 1)
        )
        uniforms = torch.rand_like(probabilities)
        mask = uniforms < probabilities
        z_out = torch.where(mask.unsqueeze(1), z_new, z)

        return z_out, self.target.log_prob(z) - self.target.log_prob(z_out)

    def inverse(self, z):
        return self.forward(z)

    def gradlogP(self, z):
        z_ = z.detach().requires_grad_()
        logp = self.target.log_prob(z_)
        grad = torch.autograd.grad(logp, z_, grad_outputs=torch.ones_like(logp))[0]
        if self.max_abs_grad:
            grad = torch.clamp(grad, max=self.max_abs_grad, min=-self.max_abs_grad)
        return grad
