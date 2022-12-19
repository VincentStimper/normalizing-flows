import numpy as np
import torch
from torch import nn

from .base import Flow


class Planar(Flow):
    """Planar flow as introduced in [arXiv: 1505.05770](https://arxiv.org/abs/1505.05770)

    ```
        f(z) = z + u * h(w * z + b)
    ```
    """

    def __init__(self, shape, act="tanh", u=None, w=None, b=None):
        """Constructor of the planar flow

        Args:
          shape: shape of the latent variable z
          h: nonlinear function h of the planar flow (see definition of f above)
          u,w,b: optional initialization for parameters
        """
        super().__init__()
        lim_w = np.sqrt(2.0 / np.prod(shape))
        lim_u = np.sqrt(2)

        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))

        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError("Nonlinearity is not implemented.")

    def forward(self, z):
        lin = torch.sum(self.w * z, list(range(1, self.w.dim())),
                        keepdim=True) + self.b
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)  # constraint w.T * u > -1
        if self.act == "tanh":
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0

        z_ = z + u * self.h(lin)
        log_det = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin.reshape(-1))))
        return z_, log_det

    def inverse(self, z):
        if self.act != "leaky_relu":
            raise NotImplementedError("This flow has no algebraic inverse.")
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        a = (lin < 0) * (
            self.h.negative_slope - 1.0
        ) + 1.0  # absorb leakyReLU slope into u
        inner = torch.sum(self.w * self.u)
        u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) \
            * self.w / torch.sum(self.w ** 2)
        dims = [-1] + (u.dim() - 1) * [1]
        u = a.reshape(*dims) * u
        inner_ = torch.sum(self.w * u, list(range(1, self.w.dim())))
        z_ = z - u * (lin / (1 + inner_)).reshape(*dims)
        log_det = -torch.log(torch.abs(1 + inner_))
        return z_, log_det
