import torch
from torch import nn


def set_requires_grad(module, flag):
    """
    Sets requires_grad flag of all parameters of a torch.nn.module
    :param module: torch.nn.module
    :param flag: Flag to set requires_grad to
    """

    for param in module.parameters():
        param.requires_grad = flag


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


class Logit():
    """
    Transform for dataloader
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    """
    def __init__(self, alpha=0):
        """
        Constructor
        :param alpha: see above
        """
        self.alpha = alpha

    def __call__(self, x):
        x_ = self.alpha + (1 - self.alpha) * x
        return torch.log(x_ / (1 - x_))


class Jitter():
    """
    Transform for dataloader
    Adds uniform jitter noise to data making sure that data stays in interval [0, 1
    """
    def __init__(self, scale=1./255):
        """
        Constructor
        :param scale: Scaling factor for noise
        """
        self.scale = scale

    def __call__(self, x):
        eps = (torch.rand_like(x) - 0.5) * self.scale
        x_ = torch.abs(x + eps)
        return x_ - 2 * torch.relu(x_ - 1)

