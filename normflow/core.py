import torch
import torch.nn as nn

class NormalizingFlow(nn.Module):
    """
    Normalizing flow model
    """
    def __init__(self, prior, decoder=None, flows=None, q0=None):
        """
        Constructor of normalizing flow model
        :param prior:
        :param decoder:
        :param flows:
        :param q0:
        """
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        if flows is None and q0 is None:
            raise NotImplementedError('Either q0 or flows must be specified.')
        self.flows = flows
        self.q0 = q0

    def forward(self, x):
