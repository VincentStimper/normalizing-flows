import torch
from torch import nn
from . import utils

class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self, layers, leaky=0.0, score_scale=None, output_fn=None,
                 output_scale=None, init_zeros=False):
        """
        :param layers: list of layer sizes from start to end
        :param leaky: slope of the leaky part of the ReLU,
        if 0.0, standard ReLU is used
        :param score_scale: Factor to apply to the scores, i.e. output before
        output_fn.
        :param output_fn: String, function to be applied to the output, either
        None, "sigmoid", "relu", "tanh", or "clampexp"
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
            if score_scale is not None:
                net.append(utils.ConstScaleLayer(score_scale))
            if output_fn is "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn is "relu":
                net.append(nn.ReLU())
            elif output_fn is "tanh":
                net.append(nn.Tanh())
            elif output_fn is "clampexp":
                net.append(utils.ClampExp())
            else:
                NotImplementedError("This output function is not implemented.")
            if output_scale is not None:
                net.append(utils.ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(self, channels, kernel_size, leaky=0.0, init_zeros=True,
                 scale_output=True, logscale_factor=3., actnorm=True,
                 weight_std=None):
        """
        Constructor
        :param channels: List of channels of conv layers, first entry is in_channels
        :param kernel_size: List of kernel sizes, same for height and width
        :param leaky: Leaky part of ReLU
        :param init_zeros: Flag whether last layer shall be initialized with zeros
        :param scale_output: Flag whether to scale output with a log scale parameter
        :param logscale_factor: Constant factor to be multiplied to log scaling
        :param actnorm: Flag whether activation normalization shall be done after
        each conv layer except output
        :param weight_std: Fixed std used to initialize every layer
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(channels[i], channels[i + 1], kernel_size[i],
                             padding=kernel_size[i] // 2, bias=(not actnorm))
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1],) + (1, 1),
                                         logscale_factor))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size[i - 1],
                             padding=kernel_size[i - 1] // 2))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)
        # Perpare variables for output scaling if needed
        self.scale_output = scale_output
        if scale_output:
            self.logs = torch.nn.Parameter(torch.zeros(1, channels[-1], 1, 1))
            self.logscale_factor = logscale_factor

    def forward(self, x):
        net_out = self.net(x)
        if self.scale_output:
            out = net_out * torch.exp(self.logs * self.logscale_factor)
        else:
            out = net_out
        return out