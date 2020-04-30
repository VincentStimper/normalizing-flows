import torch
from torch import nn

class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self, layers, leaky=0.0, sigmoid_output=False, init_zeros=True):
        """
        :param layers: list of layer sizes from start to end
        :param leaky: slope of the leaky part of the ReLU,
        if 0.0, standard ReLU is used
        :param sigmoid_output: Flag, if true, sigmoid is applied to output
        :param init_zeros: Flag, if true, weights and biases of last layer
        are initialized with zeros (helpful for deep models, see arXiv 1807.03039)
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers)-1):
            net.append(nn.Linear(layers[k], layers[k+1]))
            net.append(nn.LeakyReLU(leaky))
        net = net[:-1] # remove last ReLU
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if sigmoid_output:
            net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)