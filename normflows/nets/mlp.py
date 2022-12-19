from torch import nn
from .. import utils


class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(
        self,
        layers,
        leaky=0.0,
        score_scale=None,
        output_fn=None,
        output_scale=None,
        init_zeros=False,
        dropout=None,
    ):
        """
        layers: list of layer sizes from start to end
        leaky: slope of the leaky part of the ReLU, if 0.0, standard ReLU is used
        score_scale: Factor to apply to the scores, i.e. output before output_fn.
        output_fn: String, function to be applied to the output, either None, "sigmoid", "relu", "tanh", or "clampexp"
        output_scale: Rescale outputs if output_fn is specified, i.e. ```scale * output_fn(out / scale)```
        init_zeros: Flag, if true, weights and biases of last layer are initialized with zeros (helpful for deep models, see [arXiv 1807.03039](https://arxiv.org/abs/1807.03039))
        dropout: Float, if specified, dropout is done before last layer; if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1]))
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
