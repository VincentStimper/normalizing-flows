import torch
from torch import nn

from . import flows



class ConstScaleLayer(nn.Module):
    """
    Scaling features by a fixed factor
    """
    def __init__(self, scale=1.):
        """
        Constructor
        :param scale: Scale to apply to features
        """
        super().__init__()
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer("scale", self.scale_cpu)

    def forward(self, input):
        return input * self.scale


class ActNorm(nn.Module):
    """
    ActNorm layer with just one forward pass
    """
    def __init__(self, shape, logscale_factor=None):
        """
        Constructor
        :param shape: Same as shape in flows.ActNorm
        :param logscale_factor: Same as shape in flows.ActNorm
        """
        super().__init__()
        self.actNorm = flows.ActNorm(shape, logscale_factor=logscale_factor)

    def forward(self, input):
        out, _ = self.actNorm(input)
        return out


class ClampExp(torch.nn.Module):
    """
    Nonlinearity min(exp(lam * x), 1)
    """
    def __init__(self):
        """
        Constructor
        :param lam: Lambda parameter
        """
        super(ClampExp, self).__init__()

    def forward(self, x):
        one = torch.tensor(1., device=x.device, dtype=x.dtype)
        return torch.min(torch.exp(x), one)