import torch


def create_alternating_binary_mask(features, even=True):
    """Creates a binary mask of a given dimension which alternates its masking.

    Args:
      features: Dimension of mask.
      even: If True, even values are assigned 1s, odd 0s. If False, vice versa.

    Returns:
      Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features):
    """Creates a binary mask of a given dimension which splits its masking at the midpoint.

    Args:
      features: Dimension of mask.

    Returns:
      Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features, seed=None):
    """Creates a random binary mask of a given dimension with half of its entries randomly set to 1s.

    Args:
      features: Dimension of mask.
      seed: Seed to be used

    Returns:
      Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    if seed is None:
        generator = None
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
    indices = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False, generator=generator
    )
    mask[indices] += 1
    return mask
