import numpy as np
import torch
from torch import nn

from .base import Flow


class Radial(Flow):
    """Radial flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)

    ```
        f(z) = z + beta * h(alpha, r) * (z - z_0)
    ```
    """

    def __init__(self, shape, z_0=None):
        """Constructor of the radial flow

        Args:
          shape: shape of the latent variable z
          z_0: parameter of the radial flow
        """
        super().__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer("d", self.d_cpu)
        self.beta = nn.Parameter(torch.empty(1))
        lim = 1.0 / np.prod(shape)
        nn.init.uniform_(self.beta, -lim - 1.0, lim - 1.0)
        self.alpha = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.alpha, -lim, lim)

        if z_0 is not None:
            self.z_0 = nn.Parameter(z_0)
        else:
            self.z_0 = nn.Parameter(torch.randn(shape)[None])

    def forward(self, z):
        beta = torch.log(1 + torch.exp(self.beta)) - torch.abs(self.alpha)
        dz = z - self.z_0
        r = torch.linalg.vector_norm(dz, dim=list(range(1, self.z_0.dim())), keepdim=True)
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = -beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = z + h_arr * dz
        log_det = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        log_det = log_det.reshape(-1)
        return z_, log_det
