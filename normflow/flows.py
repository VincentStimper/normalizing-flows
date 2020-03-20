import numpy as np
import torch
import torch.nn as nn
from . import nets



# flows
class Flow(nn.Module):
    """
    Generic class for flow functions
    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')


class Planar(Flow):
    """
    Planar flow as introduced in arXiv: 1505.05770
        f(z) = z + u * h(w * z + b)
    """
    def __init__(self, shape, h=torch.tanh, u=None, w=None, b=None):
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
            self.u = nn.Parameter(torch.empty(shape)[(None,) * 2])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[(None,) * 2])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))
        self.h = h

    def forward(self, z):
        if self.h == torch.tanh:
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2)
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        else:
            raise NotImplementedError('Nonlinearity is not implemented.')
        lin = torch.sum(self.w * z, list(range(2, self.w.dim())), keepdim=True) + self.b
        z_ = z + u * self.h(lin)
        log_det = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin.squeeze())))
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
            self.z_0 = nn.Parameter(torch.randn(shape)[(None,) * 2])

    def forward(self, z):
        beta = torch.log(1 + torch.exp(self.beta)) - torch.abs(self.alpha)
        dz = z - self.z_0
        r = torch.norm(dz, dim=list(range(2, self.z_0.dim())), keepdim=True)
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = - beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = z + h_arr * dz
        log_det = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        log_det = log_det.squeeze()
        if log_det.dim() == 0:
            log_det = log_det.unsqueeze(0)
        if log_det.dim() == 1:
            log_det = log_det.unsqueeze(1)
        return z_, log_det


class AffineConstFlow(Flow):
    """ 
    scales and shifts with learned constants per dimension. In the NICE paper there is a 
    scaling layer which is a special case of this where t is None
    """
    def __init__(self, shape, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(shape)[(None,) * 2]) if scale else None
        self.t = nn.Parameter(torch.randn(shape)[(None,) * 2]) if shift else None
        
    def forward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        z_ = z * torch.exp(s) + t
        log_det = torch.sum(s, dim=2)
        return z_, log_det
    
    def inverse(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        z_ = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=2)
        print(z_.shape)
        print(log_det.shape)
        print("acf")
        return z_, log_det
       
        
class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.data_dep_init_done = False
    
    def forward(self, z):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(z.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(z * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(z)


class AffineHalfFlow(Flow):
    """
    RealNVP as introduced in arXiv: 1605.08803
    affine autoregressive flow f(z) = z * exp(s) + t
    half of the dimensions in z are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    RealNVP both scales and shifts (default), NICE only shifts
    """
    def __init__(self, shape, parity, net_class=nets.MLP, nh=4, scale=True, shift=True):
        super().__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer('d', self.d_cpu)
        self.parity = parity
        
        if scale:
            self.add_module('s_cond', net_class([self.d_cpu.numpy() // 2, nh, nh, self.d_cpu.numpy() // 2]))
            #self.s_cond.to('cuda')
        else:
            self.s_cond = lambda x: x.new_zeros(x.shape[0], x.shape[1], self.d_cpu // 2)
        if shift:
            self.add_module('t_cond', net_class([self.d_cpu.numpy() // 2, nh, nh, self.d_cpu.numpy() // 2]))
            #self.t_cond.to('cuda')
        else:
            self.t_cond = lambda x: x.new_zeros(x.shape[0], x.shape[1], self.d_cpu // 2)
        
    def forward(self, z):
        z0, z1 = z[:, :, ::2], z[:, :, 1::2]
        bs = z.shape[0]
        if self.parity:
            z0, z1 = z1, z0
        # reshape because nn.Linear takes the form (batch, dim) while we have (batch, samples, dim)
        s = self.s_cond(z0.reshape(-1, self.d_cpu // 2)).reshape(bs, -1, self.d_cpu // 2)
        t = self.t_cond(z0.reshape(-1, self.d_cpu // 2)).reshape(bs, -1, self.d_cpu // 2)
        z_0 = z0 # untouched
        z_1 = torch.exp(s) * z1 + t
        if self.parity:
            z_0, z_1 = z_1, z_0
        z_ = torch.cat([z_0, z_1], dim=2)
        log_det = torch.sum(s, dim=2)
        print(z_.shape)
        print(log_det.shape)
        print("ahf")
        return z_, log_det
    
    def inverse(self, z):
        z0, z1 = z[:, :, ::2], z[:, :, 1::2]
        bs = z.shape[0]
        if self.parity:
            z0, z1 = z1, z0
        # reshape because nn.Linear takes the form (batch, dim) while we have (batch, samples, dim)
        s = self.s_cond(z0.reshape(-1, self.d_cpu // 2)).reshape(bs, -1, self.d_cpu // 2)
        t = self.t_cond(z0.reshape(-1, self.d_cpu // 2)).reshape(bs, -1, self.d_cpu // 2)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=2)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class MaskedAffineFlow(Flow):
    """
    RealNVP as introduced in arXiv: 1605.08803
    Masked affine autoregressive flow f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    """
    def __init__(self, b, s, t):
        """
        Constructor
        :param b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
        :param s: scale mapping, i.e. neural network, where first input dimension is batch dim
        :param t: translation mapping, i.e. neural network, where first input dimension is batch dim
        """
        super().__init__()
        self.b_cpu = b.view(1, 1, *b.size())
        self.register_buffer('b', self.b_cpu)
        self.s = s
        self.t = t

    def forward(self, z):
        z_size = z.size()
        z_masked = self.b * z
        z_bd_flatten = z_masked.view(-1, *z_size[2:])
        scale = self.s(z_bd_flatten).view(*z_size)
        trans = self.t(z_bd_flatten).view(*z_size)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(2, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_size = z.size()
        z_masked = self.b * z
        z_bd_flatten = z_masked.view(-1, *z_size[2:])
        scale = self.s(z_bd_flatten).view(*z_size)
        trans = self.t(z_bd_flatten).view(*z_size)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        return z_

    
class Invertible1x1Conv(Flow):
    def __init__(self, shape):
        super().__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer('d', self.d_cpu)
        Q = torch.nn.init.orthogonal_(torch.randn(self.d_cpu, self.d_cpu))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.d_cpu))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, z):
        W = self._assemble_W().float().to(z.device)
        z_ = z @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z_, log_det

    def inverse(self, z):
        W = self._assemble_W().float().to(z.device)
        W_inv = torch.inverse(W)
        z_ = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        print(z_.shape)
        print(log_det.shape)
        print("1x1")
        return z_, log_det


    
class Glow(Flow):
    """
    Glow: Generative Flow with Invertible 1×1 Convolutions, arXiv: 1807.03039
    It has a multi-scale architecture, each flow layer consists of three parts
    ActNorm(dim=2)
    Invertible1x1Conv(dim=2)
    AffineHalfFlow(dim=2, parity=i%2, nh=32)
    """
    def __init__(self, shape, parity):
        """
        :param shape: shape of the latent variable z
        """
        super().__init__()
        self.flows = [ActNorm(shape), Invertible1x1Conv(shape), AffineHalfFlow(shape, parity)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], z.shape[1], device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot -= log_det
        return z, log_det_tot


    """
    NICE as introduced in arXiv: 1410.8516
    AffineHalfFlow(dim=2, parity=i%2, scale=False)
    added a permutation of components of z for expressivity
    """
