import torch
import numpy as np


def bitsPerDim(model, x, y=None, trans="logit", trans_param=[0.05]):
    """Computes the bits per dim for a batch of data

    Args:
      model: Model to compute bits per dim for
      x: Batch of data
      y: Class labels for batch of data if base distribution is class conditional
      trans: Transformation to be applied to images during training
      trans_param: List of parameters of the transformation

    Returns:
      Bits per dim for data batch under model
    """
    dims = torch.prod(torch.tensor(x.size()[1:]))
    if trans == "logit":
        if y is None:
            log_q = model.log_prob(x)
        else:
            log_q = model.log_prob(x, y)
        sum_dims = list(range(1, x.dim()))
        ls = torch.nn.LogSigmoid()
        sig_ = torch.sum(ls(x) / np.log(2), sum_dims)
        sig_ += torch.sum(ls(-x) / np.log(2), sum_dims)
        b = -log_q / dims / np.log(2) - np.log2(1 - trans_param[0]) + 8
        b += sig_ / dims
    else:
        raise NotImplementedError(
            "The transformation " + trans + " is not implemented."
        )
    return b


def bitsPerDimDataset(
    model, data_loader, class_cond=True, trans="logit", trans_param=[0.05]
):
    """Computes average bits per dim for an entire dataset given by a data loader

    Args:
      model: Model to compute bits per dim for
      data_loader: Data loader of dataset
      class_cond: Flag indicating whether model is class_conditional
      trans: Transformation to be applied to images during training
      trans_param: List of parameters of the transformation

    Returns:
      Average bits per dim for dataset
    """
    n = 0
    b_cum = 0
    with torch.no_grad():
        for x, y in iter(data_loader):
            b_ = bitsPerDim(
                model, x, y.to(x.device) if class_cond else None, trans, trans_param
            )
            b_np = b_.to("cpu").numpy()
            b_cum += np.nansum(b_np)
            n += len(x) - np.sum(np.isnan(b_np))
        b = b_cum / n
    return b
