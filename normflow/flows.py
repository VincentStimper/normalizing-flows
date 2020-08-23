import numpy as np
import torch
import torch.nn as nn

from . import nets


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


# Split and merge operations

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
            nc = z.size(1)
            z1 = z[:, :nc // 2, ...]
            z2 = z[:, nc // 2:, ...]
        elif self.mode == 'channel_inv':
            nc = z.size(1)
            z2 = z[:, :nc // 2, ...]
            z1 = z[:, nc // 2:, ...]
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
            z_size = z.size()
            z1 = z.view(-1)[cb.view(-1).nonzero()].view(*z_size[:-1], -1)
            z2 = z.view(-1)[(1 - cb).view(-1).nonzero()].view(*z_size[:-1], -1)
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = torch.zeros(len(z), dtype=z.dtype, device=z.device)
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
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = torch.zeros(len(z), dtype=z.dtype, device=z.device)
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


# Affine coupling layers

class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        super().__init__()
        init = torch.zeros(shape)[None]
        if scale:
            self.s = nn.Parameter(init)
        else:
            self.register_buffer('s', init)
        if shift:
            self.t = nn.Parameter(init)
        else:
            self.register_buffer('t', init)
        self.n_dim = self.s.dim()
        self.batch_dims = (torch.tensor(self.s.shape) == 1).nonzero()[:, 0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = torch.prod(torch.tensor(z.shape)[self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s, dim=list(range(1, self.n_dim)))
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = torch.prod(torch.tensor(z.shape)[self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s, dim=list(range(1, self.n_dim)))
        return z_, log_det


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(False)
        self.register_buffer('data_dep_init_done', self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            self.s.data = (-torch.log(z.std(dim=self.batch_dims, keepdim=True))).data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(True)
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            self.s.data = torch.log(z.std(dim=self.batch_dims, keepdim=True)).data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(True)
        return super().inverse(z)


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
            nc = param.size(1) // 2
            assert nc == z2.size(1)
            shift = param[:, :nc, ...]
            scale_ = param[:, nc:, ...]
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
            log_det = torch.zeros(len(z1), dtype=z1.dtype, device=z1.device)
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            nc = param.size(1) // 2
            assert nc == z2.size(1)
            shift = param[:, :nc, ...]
            scale_ = param[:, nc:, ...]
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
            log_det = torch.zeros(len(z1), dtype=z1.dtype, device=z1.device)
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

    
class Invertible1x1Conv(Flow):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        Q = torch.nn.init.orthogonal_(torch.randn(self.num_channels, self.num_channels))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.register_buffer('P', P) # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.num_channels,
                                                                    device=self.P.device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, z):
        # Permute dimensions so channel dim is last and gets used for matmul
        n_dims = z.dim()
        perm = [0] + list(range(2, n_dims)) + [1]
        perm_inv = [0, n_dims - 1] + list(range(1, n_dims - 1))
        W = self._assemble_W()
        z_ = (z.permute(*perm) @ W).permute(*perm_inv)
        log_det = torch.sum(torch.log(torch.abs(self.S)), dim=0, keepdim=True)
        if n_dims > 2:
            log_det = log_det * torch.prod(torch.tensor(z.shape[2:]))
        return z_, log_det

    def inverse(self, z):
        # Permute dimensions so channel dim is last and gets used for matmul
        n_dims = z.dim()
        perm = [0] + list(range(2, n_dims)) + [1]
        perm_inv = [0, z.dim() - 1] + list(range(1, n_dims - 1))
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        z_ = (z.permute(*perm) @ W_inv).permute(*perm_inv)
        log_det = -torch.sum(torch.log(torch.abs(self.S)), dim=0, keepdim=True)
        if n_dims > 2:
            log_det = log_det * torch.prod(torch.tensor(z.shape[2:]))
        return z_, log_det


class GlowBlock(Flow):
    """
    Glow: Generative Flow with Invertible 1×1 Convolutions, arXiv: 1807.03039
    One Block of the Glow model, comprised of
    MaskedAffineFlow (affine coupling layer
    Invertible1x1Conv (dropped if there is only one channel)
    ActNorm (first batch used for initialization)
    """
    def __init__(self, shape, channels, kernel_size, leaky=0.0, init_zeros=True,
                 coupling='affine'):
        """
        :param shape: Shape of input data as tuple of ints in form CHW
        :param channels: List of number of channels of ConvNet2d
        :param kernel_size: List of kernel sizes of ConvNet2d
        :param leaky: Leaky parameter of LeakyReLUs of ConvNet2d
        :param init_zeros: Flag whether to initialize last conv layer with zeros
        :param coupling: String, type of coupling, can be affine or additive
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Checkerboard mask
        m = [1 if i % 2 == 0 else 0 for i in range(shape[2])]
        m_ = [0 if i % 2 == 0 else 1 for i in range(shape[2])]
        mm = [m if i % 2 == 0 else m_ for i in range(shape[1])]
        mm_ = [m_ if i % 2 == 0 else m for i in range(shape[1])]
        b = torch.tensor([mm if i % 2 == 0 else mm_ for i in range(shape[0])])
        # Coupling layers
        t = nets.ConvNet2d(channels, kernel_size, leaky, init_zeros)
        if coupling == 'affine':
            s = nets.ConvNet2d(channels, kernel_size, leaky, init_zeros)
            self.flows += [MaskedAffineFlow(b, s, t)]
        elif coupling == 'additive':
            self.flows += [MaskedAffineFlow(b, None, t, scale=False)]
        else:
            raise NotImplementedError('This coupling type is not implemented.')
        t = nets.ConvNet2d(channels, kernel_size, leaky, init_zeros)
        if coupling == 'affine':
            s = nets.ConvNet2d(channels, kernel_size, leaky, init_zeros)
            self.flows += [MaskedAffineFlow(1 - b, s, t)]
        elif coupling == 'additive':
            self.flows += [MaskedAffineFlow(1 - b, None, t, scale=False)]
        else:
            raise NotImplementedError('This coupling type is not implemented.')
        # Invertible 1x1 convolution
        if shape[0] > 1:
            self.flows += [Invertible1x1Conv(channels[0])]
        # Activation normalization
        self.flows += [ActNorm((shape[0],) + (1, 1))]

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
