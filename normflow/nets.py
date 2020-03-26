import torch
from torch import nn

class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self, layers, sigmoid_output=False, leaky=0.0):
        """
        :param layers: list of layer sizes from start to end
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers)-1):
            net.append(nn.Linear(layers[k], layers[k+1]))
            net.append(nn.LeakyReLU(leaky))
        net = net[:-1] # remove last ReLU
        if sigmoid_output:
            net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)