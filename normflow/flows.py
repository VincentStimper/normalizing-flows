import numpy as np
import torch
import torch.nn as nn

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
        lim = np.sqrt(3. / np.prod(shape))
        
        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[(None,) * 2])
            nn.init.uniform_(self.u, -lim, lim)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[(None,) * 2])
            nn.init.uniform_(self.w, -lim, lim)
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
        :param h: nonlinear function h of the planar flow (see definition of f above)
        :param z_0,alpha,beta: parameters of the radial flow
        """
        super().__init__()
        self.d_cpu = torch.prod(torch.tensor(shape))
        self.register_buffer('d', self.d_cpu)
        self.beta = nn.Parameter(torch.empty(1))
        lim = np.sqrt(3. / np.prod(shape))
        nn.init.uniform_(self.beta, -lim, lim)
        self.alpha = nn.Parameter(1e-3 * torch.ones(1))

        if z_0 is not None:
            self.z_0 = nn.Parameter(z_0)
        else:
            self.z_0 = nn.Parameter(torch.zeros(shape)[(None,) * 2])

    def forward(self, z):
        beta = torch.log(1 + torch.exp(self.beta)) - self.abs(self.alpha)
        dz = z - self.z_0
        r = torch.norm(dz)
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = - beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = z + h_arr * dz
        log_det = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)
        if log_det.dim() == 1:
            log_det = log_det.unsqueeze(1)
        return z_, log_det
