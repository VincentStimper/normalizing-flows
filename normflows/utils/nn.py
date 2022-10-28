import torch
from torch import nn

from .. import flows


class ConstScaleLayer(nn.Module):
    """
    Scaling features by a fixed factor
    """

    def __init__(self, scale=1.0):
        """Constructor

        Args:
          scale: Scale to apply to features
        """
        super().__init__()
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer("scale", self.scale_cpu)

    def forward(self, input):
        return input * self.scale


class ActNorm(nn.Module):
    """
    ActNorm layer with just one forward pass
    """

    def __init__(self, shape, logscale_factor=None):
        """Constructor

        Args:
          shape: Same as shape in flows.ActNorm
          logscale_factor: Same as shape in flows.ActNorm
        """
        super().__init__()
        self.actNorm = flows.ActNorm(shape, logscale_factor=logscale_factor)

    def forward(self, input):
        out, _ = self.actNorm(input)
        return out


class ClampExp(nn.Module):
    """
    Nonlinearity min(exp(lam * x), 1)
    """

    def __init__(self):
        """Constructor

        Args:
          lam: Lambda parameter
        """
        super(ClampExp, self).__init__()

    def forward(self, x):
        one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return torch.min(torch.exp(x), one)


class PeriodicFeatures(nn.Module):
    """
    Converts a specified part of the input to periodic features
    """

    def __init__(self, ndim, ind, scale=1.0, bias=False, activation=None):
        """Constructor

        Args:
          ndim (int): number of dimensions
          ind (iterable): indices of input elements to convert to periodic features
          scale: Scalar or iterable, used to scale inputs before converting them to periodic features
          bias: Flag, whether to add a bias
          activation: Function or None, activation function to be applied
        """
        super(PeriodicFeatures, self).__init__()

        # Set up indices and permutations
        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer("ind", torch._cast_Long(ind))
        else:
            self.register_buffer("ind", torch.tensor(ind, dtype=torch.long))

        ind_ = []
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer("ind_", torch.tensor(ind_, dtype=torch.long))

        perm_ = torch.cat((self.ind, self.ind_))
        inv_perm_ = torch.zeros_like(perm_)
        for i in range(self.ndim):
            inv_perm_[perm_[i]] = i
        self.register_buffer("inv_perm", inv_perm_)

        self.weights = nn.Parameter(torch.ones(len(self.ind), 2))
        if torch.is_tensor(scale):
            self.register_buffer("scale", scale)
        else:
            self.scale = scale

        self.apply_bias = bias
        if self.apply_bias:
            self.bias = nn.Parameter(torch.zeros(len(self.ind)))

        if activation is None:
            self.activation = lambda input: input
        else:
            self.activation = activation

    def forward(self, inputs):
        inputs_ = inputs[..., self.ind]
        inputs_ = self.scale * inputs_
        inputs_ = self.weights[:, 0] * torch.sin(inputs_) + self.weights[
            :, 1
        ] * torch.cos(inputs_)
        if self.apply_bias:
            inputs_ = inputs_ + self.bias
        inputs_ = self.activation(inputs_)
        out = torch.cat((inputs_, inputs[..., self.ind_]), -1)
        return out[..., self.inv_perm]


def tile(x, n):
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)
