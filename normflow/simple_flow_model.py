import torch.nn as nn
import torch

class SimpleFlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        ld = 0.
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_

        return z, ld
