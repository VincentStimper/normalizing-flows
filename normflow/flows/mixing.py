import torch
from torch import nn

from .base import Flow

# Try importing Neural Spline Flow dependencies
try:
    from neural_spline_flows.nde.transforms.permutations import RandomPermutation
    from neural_spline_flows.nde.transforms.lu import LULinear
except:
    print('Warning: Dependencies for Neural Spline Flows could '
          'not be loaded. Other models can still be used.')


class Permute(Flow):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode='shuffle'):
        """
        Constructor
        :param num_channel: Number of channels
        :param mode: Mode of permuting features, can be shuffle for
        random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == 'shuffle':
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(dim=0, index=perm,
                                                       src=torch.arange(self.num_channels))
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z):
        if self.mode == 'shuffle':
            z = z[:, self.perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det

    def inverse(self, z):
        if self.mode == 'shuffle':
            z = z[:, self.inv_perm, ...]
        elif self.mode == 'swap':
            z1 = z[:, :(self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2:, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError('The mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det


class Invertible1x1Conv(Flow):
    """
    Invertible 1x1 convolution introduced in the Glow paper
    Assumes 4d input/output tensors of the form NCHW
    """

    def __init__(self, num_channels, use_lu=False):
        """
        Constructor
        :param num_channels: Number of channels of the data
        :param use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q = torch.qr(torch.randn(self.num_channels, self.num_channels))[0]
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer('P', P)  # remains fixed during optimization
            self.L = nn.Parameter(L)  # lower triangular portion
            S = U.diag()  # "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S
            self.register_buffer("eye", torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_S * torch.exp(self.log_S))
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

    def forward(self, z):
        if self.use_lu:
            W = self._assemble_W(inverse=True)
            log_det = -torch.sum(self.log_S)
        else:
            W_dtype = self.W.dtype
            if W_dtype == torch.float64:
                W = torch.inverse(self.W)
            else:
                W = torch.inverse(self.W.double()).type(W_dtype)
            W = W.view(*W.size(), 1, 1)
            log_det = -torch.slogdet(self.W)[1]
        W = W.view(self.num_channels, self.num_channels, 1, 1)
        z_ = torch.nn.functional.conv2d(z, W)
        log_det = log_det * z.size(2) * z.size(3)
        return z_, log_det

    def inverse(self, z):
        if self.use_lu:
            W = self._assemble_W()
            log_det = torch.sum(self.log_S)
        else:
            W = self.W
            log_det = torch.slogdet(self.W)[1]
        W = W.view(self.num_channels, self.num_channels, 1, 1)
        z_ = torch.nn.functional.conv2d(z, W)
        log_det = log_det * z.size(2) * z.size(3)
        return z_, log_det


class InvertibleAffine(Flow):
    """
    Invertible affine transformation without shift, i.e. one-dimensional
    version of the invertible 1x1 convolutions
    """

    def __init__(self, num_channels, use_lu=True):
        """
        Constructor
        :param num_channels: Number of channels of the data
        :param use_lu: Flag whether to parametrize weights through the
        LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q = torch.qr(torch.randn(self.num_channels, self.num_channels))[0]
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer('P', P)  # remains fixed during optimization
            self.L = nn.Parameter(L)  # lower triangular portion
            S = U.diag()  # "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S
            self.register_buffer("eye", torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_S * torch.exp(self.log_S))
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

    def forward(self, z):
        if self.use_lu:
            W = self._assemble_W(inverse=True)
            log_det = -torch.sum(self.log_S)
        else:
            W_dtype = self.W.dtype
            if W_dtype == torch.float64:
                W = torch.inverse(self.W)
            else:
                W = torch.inverse(self.W.double()).type(W_dtype)
            log_det = -torch.slogdet(self.W)[1]
        z_ = z @ W
        return z_, log_det

    def inverse(self, z):
        if self.use_lu:
            W = self._assemble_W()
            log_det = torch.sum(self.log_S)
        else:
            W = self.W
            log_det = torch.slogdet(self.W)[1]
        z_ = z @ W
        return z_, log_det


class LULinearPermute(Flow):
    """
    Fixed permutation combined with a linear transformation parametrized
    using the LU decomposition, used in https://arxiv.org/abs/1906.04032
    """
    def __init__(self, num_channels, identity_init=True, reverse=True):
        """
        Constructor
        :param num_channels: Number of dimensions of the data
        :param identity_init: Flag, whether to initialize linear
        transform as identity matrix
        :param reverse: Flag, change forward and inverse transform
        """
        # Initialize
        super().__init__()
        self.reverse = reverse

        # Define modules
        self.permutation = RandomPermutation(num_channels)
        self.linear = LULinear(num_channels, identity_init=identity_init)

    def forward(self, z):
        if self.reverse:
            z, log_det = self.linear.inverse(z)
            z, _ = self.permutation.inverse(z)
        else:
            z, _ = self.permutation(z)
            z, log_det = self.linear(z)
        return z, log_det.view(-1)

    def inverse(self, z):
        if self.reverse:
            z, _ = self.permutation(z)
            z, log_det = self.linear(z)
        else:
            z, log_det = self.linear.inverse(z)
            z, _ = self.permutation.inverse(z)
        return z, log_det.view(-1)
