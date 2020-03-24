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
        s = self.s if self.s is not None else torch.zeros(z.shape, device=z.device)
        t = self.t if self.t is not None else torch.zeros(z.shape, device=z.device)
        z_ = z * torch.exp(s) + t
        log_det = torch.sum(s, dim=2)
        return z_, log_det
    
    def inverse(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.shape, device=z.device)
        t = self.t if self.t is not None else z.new_zeros(z.shape, device=z.device)
        z_ = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=2)
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
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(z.std(dim=0, keepdim=True))).data
            self.t.data = (-(z * torch.exp(self.s)).mean(dim=0, keepdim=True)).data
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
        else:
            self.s_cond = lambda x: x.new_zeros(x.shape[0], x.shape[1], self.d_cpu // 2)
        
        if shift:
            self.add_module('t_cond', net_class([self.d_cpu.numpy() // 2, nh, nh, self.d_cpu.numpy() // 2]))
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
        #print(t.max())
        if self.parity:
            z_0, z_1 = z_1, z_0
        z_ = torch.cat([z_0, z_1], dim=2)
        log_det = torch.sum(s, dim=2)
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
        self.register_buffer('P', P)
        #self.P = P # remains fixed during optimization
        self.L = nn.Parameter(L) # lower triangular portion
        self.S = nn.Parameter(U.diag()) # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1)) # "crop out" diagonal, stored in S

    def _assemble_W(self):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.d, device=self.d.device))
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
        Need to reinitialize flows for Glow each training session as ActNorm initializes on first batch
        """
        super().__init__()
        self.flows = nn.ModuleList([ActNorm(shape), Invertible1x1Conv(shape), AffineHalfFlow(shape, parity)])

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

    
class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs

        self.h = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2

        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)

        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):

        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.h = nn.Tanh()

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class IAF(nn.Module):
    """
    PyTorch implementation of inverse autoregressive flows as presented in
    "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans,
    Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.
    Inverse Autoregressive Flow with either MADE MLPs or Pixel CNNs. Contains several flows. Each transformation
     takes as an input the previous stochastic z, and a context h. The structure of each flow is then as follows:
     z <- autoregressive_layer(z) + h, allow for diagonal connections
     z <- autoregressive_layer(z), allow for diagonal connections
     :
     z <- autoregressive_layer(z), do not allow for diagonal connections.
     Note that the size of h needs to be the same as h_size, which is the width of the MADE layers.
     """

    def __init__(self, z_size, num_flows=2, num_hidden=0, h_size=50, forget_bias=1., conv2d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        # self.activation = torch.nn.ReLU

        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []

        # For reordering z after each flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())

            if torch.cuda.is_available():
                z_feats = z_feats.cuda()
                zh_feats = zh_feats.cuda()
                linear_mean = linear_mean.cuda()
                linear_std = linear_std.cuda()
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))

        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):

        logdets = 0.
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                # reverse ordering to help mixing
                z = z[:, self.flip_idx]

            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = F.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)
        return z, logdets