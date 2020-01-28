import numpy as np
import torch
import torch.nn as nn

class flow(nn.Module):
    """
    Generic class for flow functions
    """
    def __index__(self):
        super.__init__()

    def forward(self, z):
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')
