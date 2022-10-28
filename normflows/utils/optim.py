from ..nets.lipschitz import InducedNormLinear, InducedNormConv2d


def set_requires_grad(module, flag):
    """Sets requires_grad flag of all parameters of a torch.nn.module

    Args:
      module: torch.nn.module
      flag: Flag to set requires_grad to
    """

    for param in module.parameters():
        param.requires_grad = flag


def clear_grad(model):
    """Set gradients of model parameter to None as this speeds up training,

    See [youtube](https://www.youtube.com/watch?v=9mS1fIYj1So)

    Args:
      model: Model to clear gradients of
    """
    for param in model.parameters():
        param.grad = None


def update_lipschitz(model, n_iterations):
    for m in model.modules():
        if isinstance(m, InducedNormConv2d) or isinstance(m, InducedNormLinear):
            m.compute_weight(update=True, n_iterations=n_iterations)
