import numpy as np
import torch
import torch.nn as nn


### Flow module ###
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

        

### Normalizing flows ###
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
        :param batch_shape: tuple of batch size and sample number
        """
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer('b', self.b_cpu)
        
        if scale:
            self.add_module('s', s)
        else:
            self.s = lambda x: x.new_zeros(*x.size())
        
        if shift:
            self.add_module('t', t)
        else:
            self.t = lambda x: x.new_zeros(*x.size())

    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        trans = self.t(z_masked)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        trans = self.t(z_masked)
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
    def __init__(self, shape):
        super().__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        Q = torch.nn.init.orthogonal_(torch.randn(self.d_cpu, self.d_cpu))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.register_buffer('P', P) # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.d_cpu, device=self.P.device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W.float()

    def forward(self, z):
        W = self._assemble_W()
        z_ = z @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z_, log_det

    def inverse(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        z_ = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return z_, log_det

class AffineConstFlow(Flow):
    """ 
    scales and shifts with learned constants per dimension. In the NICE paper there is a 
    scaling layer which is a special case of this where t is None
    """
    def __init__(self, shape, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(shape)[None]) if scale else None
        self.t = nn.Parameter(torch.zeros(shape)[None]) if shift else None
        self.n_dim = self.s.dim()
        
    def forward(self, z):
        s = self.s if self.s is not None else torch.zeros(z.shape, device=z.device)
        t = self.t if self.t is not None else torch.zeros(z.shape, device=z.device)
        z_ = z * torch.exp(s) + t
        log_det = torch.sum(s, dim=list(range(1, self.n_dim)))
        return z_, log_det
    
    def inverse(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.shape, device=z.device)
        t = self.t if self.t is not None else z.new_zeros(z.shape, device=z.device)
        z_ = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=list(range(1, self.n_dim)))
        return z_, log_det
       
        
class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            self.s.data = (-torch.log(z.std(dim=0, keepdim=True))).data
            self.t.data = (-z.mean(dim=0, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = True
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            self.s.data = torch.log(z.std(dim=0, keepdim=True)).data
            self.t.data = z.mean(dim=0, keepdim=True).data
            self.data_dep_init_done = True
        return super().inverse(z)
    
class Glow(Flow):
    """
    Glow: Generative Flow with Invertible 1×1 Convolutions, arXiv: 1807.03039
    It has a multi-scale architecture, each flow layer consists of three parts
    ActNorm (first batch used for initialization)
    Invertible1x1Conv (linear transform of variables)
    AffineHalfFlow (checkboard pattern of bit mask)
    """
    def __init__(self, b, s, t):
        """
        :param shape: shape of the latent variable z
        Need to reinitialize flows for Glow each training session as ActNorm initializes on first batch
        """
        super().__init__()
        self.flows = nn.ModuleList([ActNorm(b.size()), Invertible1x1Conv(b.size()), MaskedAffineFlow(b, s, t)])

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], z.shape[1], device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], z.shape[1], device=z.device)
        for flow in self.flows:
            z, log_det = flow.inverse(z)
            log_det_tot += log_det
        return z, log_det_tot
