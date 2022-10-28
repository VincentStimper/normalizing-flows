import torch

from .base import Flow


class PeriodicWrap(Flow):
    """
    Map periodic coordinates to fixed interval
    """

    def __init__(self, ind, bound=1.0):
        """Constructor

        ind: Iterable, indices of coordinates to be mapped
        bound: Float or iterable, bound of interval
        """
        super().__init__()
        self.ind = ind
        if torch.is_tensor(bound):
            self.register_buffer("bound", bound)
        else:
            self.bound = bound

    def forward(self, z):
        return z, torch.zeros(len(z), dtype=z.dtype, device=z.device)

    def inverse(self, z):
        z_ = z.clone()
        z_[..., self.ind] = (
            torch.remainder(z_[..., self.ind] + self.bound, 2 * self.bound) - self.bound
        )
        return z_, torch.zeros(len(z), dtype=z.dtype, device=z.device)


class PeriodicShift(Flow):
    """
    Shift and wrap periodic coordinates
    """

    def __init__(self, ind, bound=1.0, shift=0.0):
        """Constructor

        Args:
          ind: Iterable, indices of coordinates to be mapped
          bound: Float or iterable, bound of interval
          shift: Tensor, shift to be applied
        """
        super().__init__()
        self.ind = ind
        if torch.is_tensor(bound):
            self.register_buffer("bound", bound)
        else:
            self.bound = bound
        if torch.is_tensor(shift):
            self.register_buffer("shift", shift)
        else:
            self.shift = shift

    def forward(self, z):
        z_ = z.clone()
        z_[..., self.ind] = (
            torch.remainder(z_[..., self.ind] + self.shift + self.bound, 2 * self.bound)
            - self.bound
        )
        return z_, torch.zeros(len(z), dtype=z.dtype, device=z.device)

    def inverse(self, z):
        z_ = z.clone()
        z_[..., self.ind] = (
            torch.remainder(z_[..., self.ind] - self.shift + self.bound, 2 * self.bound)
            - self.bound
        )
        return z_, torch.zeros(len(z), dtype=z.dtype, device=z.device)
