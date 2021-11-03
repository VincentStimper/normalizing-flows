import torch
from torch import nn
from . import utils

# Try importing ResNet dependencies
try:
    from residual_flows.layers.base import Swish, InducedNormLinear, InducedNormConv2d
except:
    print('Warning: Dependencies for Residual Networks could '
          'not be loaded. Other models can still be used.')


class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(self, layers, leaky=0.0, score_scale=None, output_fn=None,
                 output_scale=None, init_zeros=False, dropout=None):
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
        :param dropout: Float, if specified, dropout is done before last layer;
        if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers)-2):
            net.append(nn.Linear(layers[k], layers[k+1]))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if output_fn is not None:
            if score_scale is not None:
                net.append(utils.ConstScaleLayer(score_scale))
            if output_fn == "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn == "relu":
                net.append(nn.ReLU())
            elif output_fn == "tanh":
                net.append(nn.Tanh())
            elif output_fn == "clampexp":
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
                 actnorm=False, weight_std=None):
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
                net.append(utils.ActNorm((channels[i + 1],) + (1, 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size[i - 1],
                             padding=kernel_size[i - 1] // 2))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# Lipschitz continuous neural nets for residual flow

class LipschitzMLP(nn.Module):
    """
    Fully connected neural net which is Lipschitz continuous
    with Lipschitz constant L < 1
    """
    def __init__(self, channels, lipschitz_const=0.97, max_lipschitz_iter=5,
                 lipschitz_tolerance=None, init_zeros=True):
        """
        Constructor
        :param channels: Integer list with the number of channels of
        the layers
        :param lipschitz_const: Maximum Lipschitz constant of each layer
        :param max_lipschitz_iter: Maximum number of iterations used to
        ensure that layers are Lipschitz continuous with L smaller than
        set maximum; if None, tolerance is used
        :param lipschitz_tolerance: Float, tolerance used to ensure
        Lipschitz continuity if max_lipschitz_iter is None, typically 1e-3
        :param init_zeros: Flag, whether to initialize last layer
        approximately with zeros
        """
        super().__init__()

        self.n_layers = len(channels) - 1
        self.channels = channels
        self.lipschitz_const = lipschitz_const
        self.max_lipschitz_iter = max_lipschitz_iter
        self.lipschitz_tolerance = lipschitz_tolerance
        self.init_zeros = init_zeros

        layers = []
        for i in range(self.n_layers):
            layers += [Swish(),
                InducedNormLinear(in_features=channels[i],
                    out_features=channels[i + 1], coeff=lipschitz_const,
                    domain=2, codomain=2, n_iterations=max_lipschitz_iter,
                    atol=lipschitz_tolerance, rtol=lipschitz_tolerance,
                    zero_init=init_zeros if i == (self.n_layers - 1) else False)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LipschitzCNN(nn.Module):
    """
    Convolutional neural network which is Lipschitz continuous
    with Lipschitz constant L < 1
    """
    def __init__(self, channels, kernel_size, lipschitz_const=0.97,
                 max_lipschitz_iter=5, lipschitz_tolerance=None,
                 init_zeros=True):
        """
        Constructor
        :param channels: Integer list with the number of channels of
        the layers
        :param kernel_size: Integer list of kernel sizes of the layers
        :param lipschitz_const: Maximum Lipschitz constant of each layer
        :param max_lipschitz_iter: Maximum number of iterations used to
        ensure that layers are Lipschitz continuous with L smaller than
        set maximum; if None, tolerance is used
        :param lipschitz_tolerance: Float, tolerance used to ensure
        Lipschitz continuity if max_lipschitz_iter is None, typically 1e-3
        :param init_zeros: Flag, whether to initialize last layer
        approximately with zeros
        """
        super().__init__()

        self.n_layers = len(kernel_size)
        self.channels = channels
        self.kernel_size = kernel_size
        self.lipschitz_const = lipschitz_const
        self.max_lipschitz_iter = max_lipschitz_iter
        self.lipschitz_tolerance = lipschitz_tolerance
        self.init_zeros = init_zeros

        layers = []
        for i in range(self.n_layers):
            layers += [Swish(),
                InducedNormConv2d(in_channels=channels[i],
                    out_channels=channels[i + 1], kernel_size=kernel_size[i],
                    stride=1, padding=kernel_size[i] // 2, bias=True,
                    coeff=lipschitz_const, domain=2, codomain=2,
                    n_iterations=max_lipschitz_iter, atol=lipschitz_tolerance,
                    rtol=lipschitz_tolerance,
                    zero_init=init_zeros if i == self.n_layers - 1 else False)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)