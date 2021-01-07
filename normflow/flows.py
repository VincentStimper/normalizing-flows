import numpy as np
import torch
import torch.nn as nn

from . import nets

# Try importing Resflow dependencies
try:
    from residual_flows.layers import iResBlock
except:
    print('Warning: Dependencies for Residual Flows could '
          'not be loaded. Other models can still be used.')


# Flow module
class Flow(nn.Module):
    """
    Generic class for flow functions
    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        :param z: input variable, first dimension is batch dim
        :return: transformed z and log of absolute determinant
        """
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')

        

# Normalizing flows

# Flows introduced in arXiv: 1505.05770

class Planar(Flow):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """
    def __init__(self, shape, act="tanh", u=None, w=None, b=None):
        """
        Constructor of the planar flow
        :param shape: shape of the latent variable z
        :param h: nonlinear function h of the planar flow (see definition of f above)
        :param u,w,b: optional initialization for parameters
        """
        super().__init__()
        lim_w = np.sqrt(2. / np.prod(shape))
        lim_u = np.sqrt(2)
        
        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))
        
        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError('Nonlinearity is not implemented.')

    def forward(self, z):
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        if self.act == "tanh":
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2)
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2) # constraint w.T * u neq -1, use >
            h_ = lambda x: (x<0)*(self.h.negative_slope - 1.0) + 1.0
        
        z_ = z + u * self.h(lin.unsqueeze(1))
        log_det = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin)))
        return z_, log_det
    
    def inverse(self, z):
        if self.act != "leaky_relu":
            raise NotImplementedError('This flow has no algebraic inverse.')
        lin = torch.sum(self.w * z, list(range(2, self.w.dim())), keepdim=True) + self.b
        inner = torch.sum(self.w * self.u)
        a = ((lin + self.b)/(1 + inner) < 0) * (self.h.negative_slope - 1.0) + 1.0 # absorb leakyReLU slope into u
        u = a * (self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2))
        z_ = z - 1/(1+inner) * (lin + u*self.b)
        log_det = -torch.log(torch.abs(1 + torch.sum(self.w * u)))
        if log_det.dim() == 0:
            log_det = log_det.unsqueeze(0)
        if log_det.dim() == 1:
            log_det = log_det.unsqueeze(1)
        return z_, log_det

class Radial(Flow):
    """
    Radial flow as introduced in arXiv: 1505.05770
        f(z) = z + beta * h(alpha, r) * (z - z_0)
    """
    def __init__(self, shape, z_0=None):
        """
        Constructor of the radial flow
        :param shape: shape of the latent variable z
        :param z_0: parameter of the radial flow
        """
        super().__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer('d', self.d_cpu)
        self.beta = nn.Parameter(torch.empty(1))
        lim = 1.0 / np.prod(shape)
        nn.init.uniform_(self.beta, -lim - 1.0, lim - 1.0)
        self.alpha = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.alpha, -lim, lim)

        if z_0 is not None:
            self.z_0 = nn.Parameter(z_0)
        else:
            self.z_0 = nn.Parameter(torch.randn(shape)[None])

    def forward(self, z):
        beta = torch.log(1 + torch.exp(self.beta)) - torch.abs(self.alpha)
        dz = z - self.z_0
        r = torch.norm(dz, dim=list(range(1, self.z_0.dim())))
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = - beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = z + h_arr.unsqueeze(1) * dz
        log_det = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        return z_, log_det


# Split, merge, and squeeze operations

class Split(Flow):
    """
    Split features into two sets
    """
    def __init__(self, mode='channel'):
        """
        Constructor
        :param mode: Splitting mode, can be
            channel: Splits first feature dimension, usually channels, into two halfs
            channel_inv: Same as channel, but with z1 and z2 flipped
            checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
            checkerboard_inv: Same as checkerboard, but with inverted coloring
        """
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == 'channel':
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == 'channel_inv':
            z2, z1 = z.chunk(2, dim=1)
        elif 'checkerboard' in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(*z_size[:-1], -1)
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == 'channel':
            z = torch.cat([z1, z2], 1)
        elif self.mode == 'channel_inv':
            z = torch.cat([z2, z1], 1)
        elif 'checkerboard' in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """
    def __init__(self, mode='channel'):
        super().__init__(mode)

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)


class Squeeze(Flow):
    """
    Squeeze operation of multi-scale architecture, RealNVP or Glow paper
    """
    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def forward(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det

    def inverse(self, z):
        log_det = 0
        s = z.size()
        z = z.view(*s[:2], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, log_det



# Affine coupling layers

class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param logscale_factor: Optional factor which can be used to control
        the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('s', torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('t', torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.)
        self.register_buffer('data_dep_init_done', self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.)
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.)
        return super().inverse(z)


class CCAffineConst(Flow):
    """
    Affine constant flow layer with class-conditional parameters
    """
    def __init__(self, shape, num_classes):
        super().__init__()
        self.shape = shape
        self.s = nn.Parameter(torch.zeros(shape)[None])
        self.t = nn.Parameter(torch.zeros(shape)[None])
        self.s_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.t_cc = nn.Parameter(torch.zeros(num_classes, np.prod(shape)))
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = z * torch.exp(s) + t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det

    def inverse(self, z, y):
        s = self.s + (y @ self.s_cc).view(-1, *self.shape)
        t = self.t + (y @ self.t_cc).view(-1, *self.shape)
        z_ = (z - t) * torch.exp(-s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det


class AffineCoupling(Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map='exp'):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        """
        super().__init__()
        self.add_module('param_map', param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z):
        """
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 += param
            log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 -= param
            log_det = 0
        return [z1, z2], log_det


class MaskedAffineFlow(Flow):
    """
    RealNVP as introduced in arXiv: 1605.08803
    Masked affine autoregressive flow f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    NICE is AffineFlow with only shifts (volume preserving)
    """
    def __init__(self, b, s, t, shift=True, scale=True):
        """
        Constructor
        :param b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
        :param s: scale mapping, i.e. neural network, where first input dimension is batch dim
        :param t: translation mapping, i.e. neural network, where first input dimension is batch dim
        :param shift: Flag whether shift shall be applied
        :param scale: Flag whether scale shall be applied
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer('b', self.b_cpu)
        
        if scale:
            self.add_module('s', s)
        else:
            self.s = lambda x: torch.zeros_like(x)
        
        if shift:
            self.add_module('t', t)
        else:
            self.t = lambda x: torch.zeros_like(x)

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.t(z_masked)
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


class BatchNorm(Flow):
    """
    Batch Normalization with out considering the derivatives of the batch statistics, see arXiv: 1605.08803
    """
    def __init__(self, eps=1.e-10):
        super().__init__()
        self.eps_cpu = torch.tensor(eps)
        self.register_buffer('eps', self.eps_cpu)

    def forward(self, z):
        """
        Do batch norm over batch and sample dimension
        """
        mean = torch.mean(z, dim=0, keepdims=True)
        std = torch.std(z, dim=0, keepdims=True)
        z_ = (z - mean) / torch.sqrt(std ** 2 + self.eps)
        log_det = torch.log(1 / torch.prod(torch.sqrt(std ** 2 + self.eps))).repeat(z.size()[0])
        return z_, log_det


# Residual flow layers

class Residual(Flow):
    """
    Invertible residual net block, wrapper to the implementation of Chen et al.,
    see https://github.com/rtqichen/residual-flows
    """
    def __init__(self, net, n_exact_terms=2, n_samples=1, reduce_memory=True,
                 reverse=True):
        """
        Constructor
        :param net: Neural network, must be Lipschitz continuous with L < 1
        :param n_exact_terms: Number of terms always included in the power series
        :param n_samples: Number of samples used to estimate power series
        :param reduce_memory: Flag, if true Neumann series and precomputations
        for backward pass in forward pass are done
        :param reverse: Flag, if true the map f(x) = x + net(x) is applied in
        the inverse pass, otherwise it is done in forward
        """
        super().__init__()
        self.reverse = reverse
        self.iresblock = iResBlock(net, n_samples=n_samples,
                                   n_exact_terms=n_exact_terms,
                                   neumann_grad=reduce_memory,
                                   grad_in_forward=reduce_memory)

    def forward(self, z):
        if self.reverse:
            z, log_det = self.iresblock.inverse(z, 0)
        else:
            z, log_det = self.iresblock.forward(z, 0)
        return z, -log_det.view(-1)

    def inverse(self, z):
        if self.reverse:
            z, log_det = self.iresblock.forward(z, 0)
        else:
            z, log_det = self.iresblock.inverse(z, 0)
        return z, -log_det.view(-1)

# Layers for feature/channel mixing

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
            self.register_buffer('P', P) # remains fixed during optimization
            self.L = nn.Parameter(L) # lower triangular portion
            S = U.diag()# "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S
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
            self.register_buffer('P', P) # remains fixed during optimization
            self.L = nn.Parameter(L) # lower triangular portion
            S = U.diag()# "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S
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


# Combination of flow layers

class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """
    def __init__(self, param_map, scale=True, scale_map='exp', split_mode='channel'):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot

class GlowBlock(Flow):
    """
    Glow: Generative Flow with Invertible 1×1 Convolutions, arXiv: 1807.03039
    One Block of the Glow model, comprised of
    MaskedAffineFlow (affine coupling layer
    Invertible1x1Conv (dropped if there is only one channel)
    ActNorm (first batch used for initialization)
    """
    def __init__(self, channels, hidden_channels, scale=True, scale_map='sigmoid',
                 split_mode='channel', leaky=0.0, init_zeros=True, use_lu=True,
                 net_actnorm=False):
        """
        Constructor
        :param channels: Number of channels of the data
        :param hidden_channels: number of channels in the hidden layer of the ConvNet
        :param scale: Flag, whether to include scale in affine coupling layer
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        :param leaky: Leaky parameter of LeakyReLUs of ConvNet2d
        :param init_zeros: Flag whether to initialize last conv layer with zeros
        :param use_lu: Flag whether to parametrize weights through the LU decomposition
        in invertible 1x1 convolution layers
        :param logscale_factor: Factor which can be used to control the scale of
        the log scale factor, see https://github.com/openai/glow
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Coupling layer
        kernel_size = (3, 1, 3)
        num_param = 2 if scale else 1
        if 'channel' == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif 'channel_inv' == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif 'checkerboard' in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError('Mode ' + split_mode + ' is not implemented.')
        param_map = nets.ConvNet2d(channels_, kernel_size, leaky, init_zeros,
                                   actnorm=net_actnorm)
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]
        # Invertible 1x1 convolution
        if channels > 1:
            self.flows += [Invertible1x1Conv(channels, use_lu)]
        # Activation normalization
        self.flows += [ActNorm((channels,) + (1, 1))]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


# Stochastic layers

class MetropolisHastings(Flow):
    """
    Sampling through Metropolis Hastings in Stochastic Normalizing
    Flow, see arXiv: 2002.06707
    """
    def __init__(self, dist, proposal, steps):
        """
        Constructor
        :param dist: Distribution to sample from
        :param proposal: Proposal distribution
        :param steps: Number of MCMC steps to perform
        """
        super().__init__()
        self.dist = dist
        self.proposal = proposal
        self.steps = steps

    def forward(self, z):
        # Initialize number of samples and log(det)
        num_samples = len(z)
        log_det = torch.zeros(num_samples, dtype=z.dtype, device=z.device)
        # Get log(p) for current samples
        log_p = self.dist.log_prob(z)
        for i in range(self.steps):
            # Make proposal and get log(p)
            z_, log_p_diff = self.proposal(z)
            log_p_ = self.dist.log_prob(z_)
            # Make acceptance decision
            w = torch.rand(num_samples, dtype=z.dtype, device=z.device)
            log_w_accept = log_p_ - log_p + log_p_diff
            w_accept = torch.clamp(torch.exp(log_w_accept), max=1)
            accept = w <= w_accept
            # Update samples, log(det), and log(p)
            z = torch.where(accept.unsqueeze(1), z_, z)
            log_det_ = log_p - log_p_
            log_det = torch.where(accept, log_det + log_det_, log_det)
            log_p = torch.where(accept, log_p_, log_p)
        return z, log_det

    def inverse(self, z):
        # Equivalent to forward pass
        return self.forward(z)


class HamiltonianMonteCarlo(Flow):
    """
    Flow layer using the HMC proposal in Stochastic Normalising Flows,
    see arXiv: 2002.06707
    """
    def __init__(self, target, steps, log_step_size, log_mass):
        """
        Constructor
        :param target: The stationary distribution of this Markov transition. Should be logp
        :param steps: The number of leapfrog steps
        :param log_step_size: The log step size used in the leapfrog integrator. shape (dim)
        :param log_mass: The log_mass determining the variance of the momentum samples. shape (dim)
        """
        super().__init__()
        self.target = target
        self.steps = steps
        self.register_parameter('log_step_size', torch.nn.Parameter(log_step_size))
        self.register_parameter('log_mass', torch.nn.Parameter(log_mass))

    def forward(self, z):
        # Draw momentum
        p = torch.randn_like(z) * torch.exp(0.5 * self.log_mass)

        # leapfrog
        z_new = z.clone()
        p_new = p.clone()
        step_size = torch.exp(self.log_step_size)
        for i in range(self.steps):
            p_half = p_new - (step_size/2.0) * -self.gradlogP(z_new)
            z_new = z_new + step_size * (p_half/torch.exp(self.log_mass))
            p_new = p_half - (step_size/2.0) * -self.gradlogP(z_new)

        # Metropolis Hastings correction
        probabilities = torch.exp(
            self.target.log_prob(z_new) - self.target.log_prob(z) - \
            0.5 * torch.sum(p_new ** 2 / torch.exp(self.log_mass), 1) + \
            0.5 * torch.sum(p ** 2 / torch.exp(self.log_mass), 1))
        uniforms = torch.rand_like(probabilities)
        mask = uniforms < probabilities
        z_out = torch.where(mask.unsqueeze(1), z_new, z)

        return z_out, self.target.log_prob(z) - self.target.log_prob(z_out)

    def inverse(self, z):
        return self.forward(z)

    def gradlogP(self, z):
        z_ = z.detach().requires_grad_()
        logp = self.target.log_prob(z_)
        return torch.autograd.grad(logp, z_,
            grad_outputs=torch.ones_like(logp))[0]
