import torch.nn as nn

class SimpleFlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = flows

    def forward(self, x):
        log_det_ = 0.
        for flow in self.flows:
            z, log_det = flow(x)
            log_det_ += log_det
            z_ = z
        return z_, log_det_
