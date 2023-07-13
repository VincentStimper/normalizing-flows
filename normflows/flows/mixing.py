import torch
import numpy as np
from torch import nn
from torch.nn import functional as F, init

from .base import Flow


class Permute(Flow):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode="shuffle"):
        """Constructor

        Args:
          num_channel: Number of channels
          mode: Mode of permuting features, can be shuffle for random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == "shuffle":
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(
                dim=0, index=perm, src=torch.arange(self.num_channels)
            )
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z, context=None):
        if self.mode == "shuffle":
            z = z[:, self.perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det

    def inverse(self, z, context=None):
        if self.mode == "shuffle":
            z = z[:, self.inv_perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : (self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det


class Invertible1x1Conv(Flow):
    """
    Invertible 1x1 convolution introduced in the Glow paper
    Assumes 4d input/output tensors of the form NCHW
    """

    def __init__(self, num_channels, use_lu=False):
        """Constructor

        Args:
          num_channels: Number of channels of the data
          use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q, _ = torch.linalg.qr(torch.randn(self.num_channels, self.num_channels))
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer("P", P)  # remains fixed during optimization
            self.L = nn.Parameter(L)  # lower triangular portion
            S = U.diag()  # "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(
                torch.triu(U, diagonal=1)
            )  # "crop out" diagonal, stored in S
            self.register_buffer("eye", torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(
            self.sign_S * torch.exp(self.log_S)
        )
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
        """Constructor

        Args:
          num_channels: Number of channels of the data
          use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q, _ = torch.linalg.qr(torch.randn(self.num_channels, self.num_channels))
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer("P", P)  # remains fixed during optimization
            self.L = nn.Parameter(L)  # lower triangular portion
            S = U.diag()  # "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(
                torch.triu(U, diagonal=1)
            )  # "crop out" diagonal, stored in S
            self.register_buffer("eye", torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(
            self.sign_S * torch.exp(self.log_S)
        )
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

    def forward(self, z, context=None):
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

    def inverse(self, z, context=None):
        if self.use_lu:
            W = self._assemble_W()
            log_det = torch.sum(self.log_S)
        else:
            W = self.W
            log_det = torch.slogdet(self.W)[1]
        z_ = z @ W
        return z_, log_det


"""LU Linear Permutation for Neural Spline Flows"""


class _Permutation(Flow):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError("Permutation must be a 1D tensor.")

        super().__init__()
        self._dim = dim
        self.register_buffer("_permutation", permutation)

    @property
    def _inverse_permutation(self):
        return torch.argsort(self._permutation)

    @staticmethod
    def _permute(inputs, permutation, dim):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError(
                "Dimension {} in inputs must be of size {}.".format(
                    dim, len(permutation)
                )
            )
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = torch.zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._permute(inputs, self._permutation, self._dim)

    def inverse(self, inputs, context=None):
        return self._permute(inputs, self._inverse_permutation, self._dim)


class _RandomPermutation(_Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        super().__init__(torch.randperm(features), dim)


class _LinearCache(object):
    """Helper class to store the cache of a linear transform.

    The cache consists of: the weight matrix, its inverse and its log absolute determinant.
    """

    def __init__(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None


class _Linear(Flow):
    """Abstract base class for linear transforms that parameterize a weight matrix."""

    def __init__(self, features, using_cache=False):
        super().__init__()

        self.features = features
        self.bias = nn.Parameter(torch.zeros(features))

        # Caching flag and values.
        self.using_cache = using_cache
        self.cache = _LinearCache()

    def forward(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_forward_cache()
            outputs = F.linear(inputs, self.cache.weight, self.bias)
            logabsdet = self.cache.logabsdet * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.forward_no_cache(inputs)

    def _check_forward_cache(self):
        if self.cache.weight is None and self.cache.logabsdet is None:
            self.cache.weight, self.cache.logabsdet = self.weight_and_logabsdet()

        elif self.cache.weight is None:
            self.cache.weight = self.weight()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def inverse(self, inputs, context=None):
        if not self.training and self.using_cache:
            self._check_inverse_cache()
            outputs = F.linear(inputs - self.bias, self.cache.inverse)
            logabsdet = (-self.cache.logabsdet) * torch.ones(outputs.shape[0])
            return outputs, logabsdet
        else:
            return self.inverse_no_cache(inputs)

    def _check_inverse_cache(self):
        if self.cache.inverse is None and self.cache.logabsdet is None:
            (
                self.cache.inverse,
                self.cache.logabsdet,
            ) = self.weight_inverse_and_logabsdet()

        elif self.cache.inverse is None:
            self.cache.inverse = self.weight_inverse()

        elif self.cache.logabsdet is None:
            self.cache.logabsdet = self.logabsdet()

    def train(self, mode=True):
        if mode:
            # If training again, invalidate cache.
            self.cache.invalidate()
        return super().train(mode)

    def use_cache(self, mode=True):
        self.using_cache = mode

    def weight_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # and its logabsdet together.
        return self.weight(), self.logabsdet()

    def weight_inverse_and_logabsdet(self):
        # To be overridden by subclasses if it is more efficient to compute the weight matrix
        # inverse and weight matrix logabsdet together.
        return self.weight_inverse(), self.logabsdet()

    def forward_no_cache(self, inputs):
        """Applies `forward` method without using the cache."""
        raise NotImplementedError()

    def inverse_no_cache(self, inputs):
        """Applies `inverse` method without using the cache."""
        raise NotImplementedError()

    def weight(self):
        """Returns the weight matrix."""
        raise NotImplementedError()

    def weight_inverse(self):
        """Returns the inverse weight matrix."""
        raise NotImplementedError()

    def logabsdet(self):
        """Returns the log absolute determinant of the weight matrix."""
        raise NotImplementedError()


class _LULinear(_Linear):
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, inputs):
        """
        Cost:

        ```
            output = O(D^2N)
            logabsdet = O(D)
        ```

        where:

        ```
            D = num of features
            N = num of inputs
        ```
        """
        lower, upper = self._create_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """
        Cost:

        ```
            output = O(D^2N)
            logabsdet = O(D)
        ```

        where:

        ```
            D = num of features
            N = num of inputs
        ```
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        try:
            outputs = torch.linalg.solve_triangular(
                lower, outputs.t(), upper=False, unitriangular=True
            )
            outputs = torch.linalg.solve_triangular(
                upper, outputs, upper=True, unitriangular=False
            )
        except:
            outputs, _ = torch.triangular_solve(
                outputs.t(), lower, upper=False, unitriangular=True
            )
            outputs, _ = torch.triangular_solve(
                outputs, upper, upper=True, unitriangular=False
            )
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """
        Cost:

        ```
            weight = O(D^3)
        ```

        where:

        ```
            D = num of features
        ```
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """
        Cost:

        ```
            inverse = O(D^3)
        ```

        where:

        ```
            D = num of features
        ```
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.trtrs(identity, lower, upper=False, unitriangular=True)
        weight_inverse, _ = torch.trtrs(
            lower_inverse, upper, upper=True, unitriangular=False
        )
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """
        Cost:

        ```
            logabsdet = O(D)
        ```

        where:

        ```
            D = num of features
        ```
        """
        return torch.sum(torch.log(self.upper_diag))


class LULinearPermute(Flow):
    """
    Fixed permutation combined with a linear transformation parametrized
    using the LU decomposition, used in https://arxiv.org/abs/1906.04032
    """

    def __init__(self, num_channels, identity_init=True):
        """Constructor

        Args:
          num_channels: Number of dimensions of the data
          identity_init: Flag, whether to initialize linear transform as identity matrix
        """
        # Initialize
        super().__init__()

        # Define modules
        self.permutation = _RandomPermutation(num_channels)
        self.linear = _LULinear(num_channels, identity_init=identity_init)

    def forward(self, z, context=None):
        z, log_det = self.linear.inverse(z, context=context)
        z, _ = self.permutation.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, _ = self.permutation(z, context=context)
        z, log_det = self.linear(z, context=context)
        return z, log_det.view(-1)
