import torch
from torch import nn
import numpy as np

from . import flows

# Try importing ResNet dependencies
try:
    from residual_flows.layers.base import InducedNormLinear, InducedNormConv2d
except:
    print('Warning: Dependencies for Residual Networks could '
          'not be loaded. Other models can still be used.')


def set_requires_grad(module, flag):
    """
    Sets requires_grad flag of all parameters of a torch.nn.module
    :param module: torch.nn.module
    :param flag: Flag to set requires_grad to
    """

    for param in module.parameters():
        param.requires_grad = flag


class ConstScaleLayer(nn.Module):
    """
    Scaling features by a fixed factor
    """
    def __init__(self, scale=1.):
        """
        Constructor
        :param scale: Scale to apply to features
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
        """
        Constructor
        :param shape: Same as shape in flows.ActNorm
        :param logscale_factor: Same as shape in flows.ActNorm
        """
        super().__init__()
        self.actNorm = flows.ActNorm(shape, logscale_factor=logscale_factor)

    def forward(self, input):
        out, _ = self.actNorm(input)
        return out


# Dataset transforms

class Logit():
    """
    Transform for dataloader
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    """
    def __init__(self, alpha=0):
        """
        Constructor
        :param alpha: see above
        """
        self.alpha = alpha

    def __call__(self, x):
        x_ = self.alpha + (1 - self.alpha) * x
        return torch.log(x_ / (1 - x_))

    def inverse(self, x):
        return (torch.sigmoid(x) - self.alpha) / (1 - self.alpha)


class Jitter():
    """
    Transform for dataloader
    Adds uniform jitter noise to data
    """
    def __init__(self, scale=1./256):
        """
        Constructor
        :param scale: Scaling factor for noise
        """
        self.scale = scale

    def __call__(self, x):
        eps = torch.rand_like(x) * self.scale
        x_ = x + eps
        return x_


class Scale():
    """
    Transform for dataloader
    Adds uniform jitter noise to data
    """
    def __init__(self, scale=255./256.):
        """
        Constructor
        :param scale: Scaling factor for noise
        """
        self.scale = scale

    def __call__(self, x):
        return x * self.scale


# Nonlinearities

class ClampExp(torch.nn.Module):
    """
    Nonlinearity min(exp(lam * x), 1)
    """
    def __init__(self):
        """
        Constructor
        :param lam: Lambda parameter
        """
        super(ClampExp, self).__init__()

    def forward(self, x):
        one = torch.tensor(1., device=x.device, dtype=x.dtype)
        return torch.min(torch.exp(x), one)


# Functions for model analysis

def bitsPerDim(model, x, y=None, trans='logit', trans_param=[0.05]):
    """
    Computes the bits per dim for a batch of data
    :param model: Model to compute bits per dim for
    :param x: Batch of data
    :param y: Class labels for batch of data if base distribution is class conditional
    :param trans: Transformation to be applied to images during training
    :param trans_param: List of parameters of the transformation
    :return: Bits per dim for data batch under model
    """
    dims = torch.prod(torch.tensor(x.size()[1:]))
    if trans == 'logit':
        if y is None:
            log_q = model.log_prob(x)
        else:
            log_q = model.log_prob(x, y)
        sum_dims = list(range(1, x.dim()))
        ls = torch.nn.LogSigmoid()
        sig_ = torch.sum(ls(x) / np.log(2), sum_dims)
        sig_ += torch.sum(ls(-x) / np.log(2), sum_dims)
        b = - log_q / dims / np.log(2) - np.log2(1 - trans_param[0]) + 8
        b += sig_ / dims
    else:
        raise NotImplementedError('The transformation ' + trans + ' is not implemented.')
    return b


def bitsPerDimDataset(model, data_loader, class_cond=True, trans='logit',
                      trans_param=[0.05]):
    """
    Computes average bits per dim for an entire dataset given by a data loader
    :param model: Model to compute bits per dim for
    :param data_loader: Data loader of dataset
    :param class_cond: Flag indicating whether model is class_conditional
    :param trans: Transformation to be applied to images during training
    :param trans_param: List of parameters of the transformation
    :return: Average bits per dim for dataset
    """
    n = 0
    b_cum = 0
    with torch.no_grad():
        for x, y in iter(data_loader):
            b_ = bitsPerDim(model, x, y.to(x.device) if class_cond else None,
                            trans, trans_param)
            b_np = b_.to('cpu').numpy()
            b_cum += np.nansum(b_np)
            n += len(x) - np.sum(np.isnan(b_np))
        b = b_cum / n
    return b


def clear_grad(model):
    """
    Set gradients of model parameter to None as this speeds up training,
    see https://www.youtube.com/watch?v=9mS1fIYj1So
    :param model: Model to clear gradients of
    """
    for param in model.parameters():
        param.grad = None


def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, InducedNormConv2d) or isinstance(m, InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)
