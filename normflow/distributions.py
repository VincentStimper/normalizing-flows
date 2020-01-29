import torch
import torch.nn as nn

class ParametrizedConditionalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: Variable to condition on, first dimension is batch size
        :return: sample of z for x, log probability for sample
        """
        raise NotImplementedError


class Dirac(ParametrizedConditionalDistribution):
    def __init__(self):
        super.__init__()

    def forward(self, x):
        z = x
        log_p = torch.zeros(x.size[0])
        return z, log_p