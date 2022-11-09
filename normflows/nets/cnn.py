from torch import nn
from .. import utils


class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(
        self,
        channels,
        kernel_size,
        leaky=0.0,
        init_zeros=True,
        actnorm=False,
        weight_std=None,
    ):
        """Constructor

        Args:
          channels: List of channels of conv layers, first entry is in_channels
          kernel_size: List of kernel sizes, same for height and width
          leaky: Leaky part of ReLU
          init_zeros: Flag whether last layer shall be initialized with zeros
          scale_output: Flag whether to scale output with a log scale parameter
          logscale_factor: Constant factor to be multiplied to log scaling
          actnorm: Flag whether activation normalization shall be done after each conv layer except output
          weight_std: Fixed std used to initialize every layer
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size[i],
                padding=kernel_size[i] // 2,
                bias=(not actnorm),
            )
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(utils.ActNorm((channels[i + 1],) + (1, 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(
            nn.Conv2d(
                channels[i - 1],
                channels[i],
                kernel_size[i - 1],
                padding=kernel_size[i - 1] // 2,
            )
        )
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
