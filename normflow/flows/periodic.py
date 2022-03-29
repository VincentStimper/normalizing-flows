import torch

from .base import Flow



class Periodic(Flow):
    """
    Map periodic coordinates to fixed interval
    """
    def __init__(self, ind, bound=1.):
        """
        Constructor
        :param ind: Iterable, indices of coordinates to be mapped
        :param bound: Float or iterable, bound of interval
        """
        super().__init__()
        self.ind = ind
        if torch.is_tensor(bound):
            self.register_buffer('bound', bound)
        else:
            self.bound = bound

    def forward(self, z):
        return z, torch.zeros(len(z), dtype=z.dtype, device=z.device)

    def inverse(self, z):
        z_ = z.clone()
        z_[..., self.ind] = torch.remainder(z_ + self.bound, 2 * self.bound) \
                            - self.bound
        return z_, torch.zeros(len(z), dtype=z.dtype, device=z.device)
