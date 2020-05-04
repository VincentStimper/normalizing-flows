import torch
from torch import nn
from . import utils

class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self, layers, leaky=0.0, output_fn=None, output_scale=1., init_zeros=False):
        """
        :param layers: list of layer sizes from start to end
        :param leaky: slope of the leaky part of the ReLU,
        if 0.0, standard ReLU is used
        :param output_fn: String, function to be applied to the output, either
        None, "sigmoid", "relu", or "tanh"
        :param output_scale: Rescale outputs if output_fn is specified, i.e.
        scale * output_fn(out / scale)
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
        if output_fn is not None:
            net.append(utils.ConstScaleLayer(1 / output_scale))
            if output_fn is "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn is "relu":
                net.append(nn.ReLU())
            elif output_fn is "tanh":
                net.append(nn.Tanh())
            else:
                NotImplementedError("This output function is not implemented.")
            net.append(utils.ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)