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