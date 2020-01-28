import numpy as np
import torch
import torch.nn as nn

class flow(nn.Module):
    """
    Generic class for flow functions
    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')


class planar(flow):
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
        self.u = nn.Parameter(torch.randn(shape))
        self.w = nn.Parameter(torch.randn(shape))
        self.b = nn.Parameter(torch.randn(shape))
        self.h = h

    def forward(self, z):
        if self.h == torch.tanh:
            prod = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(prod)) - 1 - prod) * self.w / torch.sum(self.w ** 2)
        else:
            raise NotImplementedError('Nonlinearity is not implemented.')
        return z + u * self.h(torch.sum(self.w * z) + self.b)
