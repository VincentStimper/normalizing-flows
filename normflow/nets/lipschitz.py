from torch import nn

# Try importing ResNet dependencies
try:
    from residual_flows.layers.base import Swish, InducedNormLinear, InducedNormConv2d
except:
    print('Warning: Dependencies for Residual Networks could '
          'not be loaded. Other models can still be used.')



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