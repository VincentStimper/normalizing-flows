import numpy as np
import torch
import torch.nn as nn

class Flow(nn.Module):
    """
    Generic class for flow functions
    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')


class Planar(Flow):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """
    def __init__(self, shape, h=torch.tanh):
        """
        Constructor of the planar flow
        :param shape: shape of the latent variable z
        :param h: nonlinear function h of the planar flow (see definition of f above)
        """
        super().__init__()
        self.u = nn.Parameter(torch.randn(shape)[(None,) * 2])
        self.w = nn.Parameter(torch.randn(shape)[(None,) * 2])
        self.b = nn.Parameter(torch.randn(1))
        self.h = h

    def forward(self, z):
        if self.h == torch.tanh:
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2)
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        else:
            raise NotImplementedError('Nonlinearity is not implemented.')
        inner = torch.sum(self.w * z, list(range(2, self.w.dim())), keepdim=True)
        z_ = z + u * self.h(inner + self.b)
        log_det = torch.abs(1 + torch.sum(self.w * u) * h_(inner.squeeze() + self.b))
        return z_, log_det
