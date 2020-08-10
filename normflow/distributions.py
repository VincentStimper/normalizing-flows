import torch
import torch.nn as nn
import numpy as np

class BaseDistribution(nn.Module):
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """
    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """
        Samples from base distribution and calculates log probability
        :param num_samples: Number of samples to draw from the distriubtion
        :return: Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """
        Calculate log probability of batch of samples
        :param z: Batch of random variables to determine log probability for
        :return: log probability for each batch element
        """
        raise NotImplementedError


class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, d, trainable=True):
        """
        Constructor
        :param d: Dimension of Gaussian distribution
        """
        super().__init__()
        self.d = d
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.d))
            self.log_scale = nn.Parameter(torch.zeros(1, self.d))
        else:
            self.register_buffer("loc", torch.zeros(1, self.d))
            self.register_buffer("log_scale", torch.zeros(1, self.d))

    def forward(self, num_samples=1):
        eps = torch.randn((num_samples, self.d), dtype=self.loc.dtype, device=self.loc.device)
        z = self.loc + torch.exp(self.log_scale) * eps
        log_p = - 0.5 * self.d * np.log(2 * np.pi) - torch.sum(self.log_scale + 0.5 * torch.pow(eps, 2), 1)
        return z, log_p

    def log_prob(self, z):
        log_p = - 0.5 * self.d * np.log(2 * np.pi)\
                - torch.sum(self.log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(self.log_scale), 2), 1)
        return log_p


class ResampledGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix,
    resampled according to a acceptance probability determined by a neural network,
    see arXiv 1810.11428
    """
    def __init__(self, d, a, T, eps, trainable=True):
        """
        Constructor
        :param d: Dimension of Gaussian distribution
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        """
        super().__init__()
        self.d = d
        self.a = a
        self.T = T
        self.eps = eps
        self.Z = None
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.d))
            self.log_scale = nn.Parameter(torch.zeros(1, self.d))
        else:
            self.register_buffer("loc", torch.zeros(1, self.d))
            self.register_buffer("log_scale", torch.zeros(1, self.d))

    def forward(self, num_samples=1):
        t = 0
        z = torch.zeros(num_samples, self.d, dtype=self.loc.dtype, device=self.loc.device)
        s = 0
        for i in range(self.T):
            eps = torch.randn((num_samples, self.d), dtype=self.loc.dtype, device=self.loc.device)
            z_ = self.loc + torch.exp(self.log_scale) * eps
            acc = self.a(z)
            dec = torch.randn_like(acc) < acc
            for i, dec_ in enumerate(dec):
                if dec_ or t == self.T:
                    z[s] = z_[i]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == self.d:
                    break
            if s == self.d:
                break
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(self.log_scale), 2), 1)
        acc = self.a(z)
        if self.Z == None:
            self.Z = torch.mean(acc).detach()
        alpha = (1 - self.Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc / self.Z + alpha) + log_p_gauss
        return z, log_p

    def log_prob(self, z):
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(self.log_scale), 2), 1)
        acc = self.a(z)
        Z_batch = torch.mean(acc)
        if self.training:
            if self.Z == None:
                self.Z = Z_batch
                Z = Z_batch
            else:
                self.Z = ((1 - self.eps) * self.Z + self.eps * Z_batch).detach()
                Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return log_p


class GaussianMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """
    def __init__(self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True):
        """
        Constructor
        :param n_modes: Number of modes of the mixture model
        :param dim: Number of dimensions of each Gaussian
        :param loc: List of mean values
        :param scale: List of diagonals of the covariance matrices
        :param weights: List of mode probabilities
        :param trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1. * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1. * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1. * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1. * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1. * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1. * weights)))

    def forward(self, num_samples=1):
        # Sample mode indices
        mode_ind = torch.randint(high=self.n_modes, size=(num_samples,))
        mode_1h = torch.zeros((num_samples, self.n_modes), dtype=torch.int64)
        mode_1h.scatter_(1, mode_ind[:, None], 1)
        mode_1h = mode_1h[..., None]

        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Get samples
        eps = torch.randn(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps * scale_sample + loc_sample

        # Compute log probability
        log_p = - 0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights)\
                - 0.5 * torch.sum(torch.pow(eps, 2), 1, keepdim=True)\
                - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = - 0.5 * self.dim * np.log(2 * np.pi) + torch.log(weights) \
                - 0.5 * torch.sum(torch.pow(eps, 2), 2) \
                - torch.sum(self.log_scale, 2)
        log_p = torch.logsumexp(log_p, 1)

        return log_p


class GaussianPCA(BaseDistribution):
    """
    Gaussian distribution resulting from linearly mapping a normal distributed latent
    variable describing the "content of the target"
    """
    def __init__(self, dim, latent_dim=None, sigma=0.1):
        """
        Constructor
        :param dim: Number of dimensions of the flow variables
        :param latent_dim: Number of dimensions of the latent "content" variable;
                           if None it is set equal to dim
        :param sigma: Noise level
        """
        super().__init__()

        self.dim = dim
        if latent_dim is None:
            self.latent_dim = dim
        else:
            self.latent_dim = latent_dim

        self.loc = nn.Parameter(torch.zeros(1, dim))
        self.W = nn.Parameter(torch.randn(latent_dim, dim))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))

    def forward(self, num_samples=1):
        eps = torch.randn(num_samples, self.latent_dim, dtype=self.loc.dtype,
                          device=self.loc.device)
        z_ = torch.matmul(eps, self.W)
        z = z_ + self.loc

        Sig = torch.matmul(self.W.T, self.W) \
              + torch.exp(self.log_sigma * 2) \
              * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = self.dim / 2 * np.log(2 * np.pi) - 0.5 * torch.det(Sig) \
                - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)

        return z, log_p

    def log_prob(self, z):
        z_ = z - self.loc

        Sig = torch.matmul(self.W.T, self.W) \
              + torch.exp(self.log_sigma * 2) \
              * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = self.dim / 2 * np.log(2 * np.pi) - 0.5 * torch.det(Sig) \
                - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)

        return log_p


class BaseEncoder(nn.Module):
    """
    Base distribution of a flow-based variational autoencoder
    Parameters of the distribution depend of the target variable x
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        """
        :param x: Variable to condition on, first dimension is batch size
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        raise NotImplementedError

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch size
        :param x: Variable to condition on, first dimension is batch size
        :return: log probability of z given x
        """
        raise NotImplementedError


class Dirac(BaseEncoder):
    def __init__(self):
        super().__init__()

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1)
        log_p = torch.zeros(z.size()[0:2])
        return z, log_p

    def log_prob(self, z, x):
        log_p = torch.zeros(z.size()[0:2])
        return log_p
    
    
class Uniform(BaseEncoder):
    def __init__(self, zmin=0.0, zmax=1.0):
        super().__init__()
        self.zmin = zmin
        self.zmax = zmax
        self.log_p = -torch.log(zmax-zmin)

    def forward(self, x, num_samples=1):
        z = x.unsqueeze(1).repeat(1, num_samples, 1).uniform_(min=self.zmin, max=self.zmax)
        log_p = torch.zeros(z.size()[0:2]).fill_(self.log_p)
        return z, log_p

    def log_prob(self, z, x):
        log_p = torch.zeros(z.size()[0:2]).fill_(self.log_p)
        return log_p


class ConstDiagGaussian(BaseEncoder):
    def __init__(self, loc, scale):
        """
        Multivariate Gaussian distribution with diagonal covariance and parameters being constant wrt x
        :param loc: mean vector of the distribution
        :param scale: vector of the standard deviations on the diagonal of the covariance matrix
        """
        super().__init__()
        self.d = len(loc)
        if not torch.is_tensor(loc):
            loc = torch.tensor(loc).float()
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale).float()
        self.loc = nn.Parameter(loc.reshape((1, 1, self.d)))
        self.scale = nn.Parameter(scale)

    def forward(self, x=None, num_samples=1):
        """
        :param x: Variable to condition on, will only be used to determine the batch size
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        if x is not None:
            batch_size = len(x)
        else:
            batch_size = 1
        eps = torch.randn((batch_size, num_samples, self.d), device=x.device)
        z = self.loc + self.scale * eps
        log_p = - 0.5 * self.d * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * torch.pow(eps, 2), 2)
        return z, log_p

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch dimension
        :param x: Variable to condition on, first dimension is batch dimension
        :return: log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        log_p = - 0.5 * self.d * np.log(2 * np.pi) - torch.sum(torch.log(self.scale) + 0.5 * ((z - self.loc) / self.scale) ** 2, 2)
        return log_p


class NNDiagGaussian(BaseEncoder):
    """
    Diagonal Gaussian distribution with mean and variance determined by a neural network
    """
    def __init__(self, net):
        """
        Constructor
        :param net: net computing mean (first n / 2 outputs), standard deviation (second n / 2 outputs)
        """
        super().__init__()
        self.net = net

    def forward(self, x, num_samples=1):
        """
        :param x: Variable to condition on
        :param num_samples: number of samples to draw per element of mini-batch
        :return: sample of z for x, log probability for sample
        """
        batch_size = len(x)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        std = torch.exp(0.5 * mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1))
        eps = torch.randn((batch_size, num_samples) + tuple(mean.size()[2:]), device=x.device)
        z = mean + std * eps
        log_p = - 0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi)\
                - torch.sum(torch.log(std) + 0.5 * torch.pow(eps, 2), list(range(2, z.dim())))
        return z, log_p

    def log_prob(self, z, x):
        """
        :param z: Primary random variable, first dimension is batch dimension
        :param x: Variable to condition on, first dimension is batch dimension
        :return: log probability of z given x
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        mean_std = self.net(x)
        n_hidden = mean_std.size()[1] // 2
        mean = mean_std[:, :n_hidden, ...].unsqueeze(1)
        var = torch.exp(mean_std[:, n_hidden:(2 * n_hidden), ...].unsqueeze(1))
        log_p = - 0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi)\
                - 0.5 * torch.sum(torch.log(var) + (z - mean) ** 2 / var, 2)
        return log_p


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        """
        Decodes z to x
        :param z: latent variable
        :return: x, std of x
        """
        raise NotImplementedError

    def log_prob(self, x, z):
        """
        :param x: observable
        :param z: latent variable
        :return: log(p) of x given z
        """
        raise NotImplementedError


class NNDiagGaussianDecoder(Decoder):
    """
    Decoder representing a diagonal Gaussian distribution with mean and std parametrized by a NN
    """
    def __init__(self, net):
        """
        Constructor
        :param net: neural network parametrizing mean and standard deviation of diagonal Gaussian
        """
        super().__init__()
        self.net = net

    def forward(self, z):
        z_size = z.size()
        mean_std = self.net(z.view(-1, *z_size[2:])).view(z_size)
        n_hidden = mean_std.size()[2] // 2
        mean = mean_std[:, :, :n_hidden, ...]
        std = torch.exp(0.5 * mean_std[:, :, n_hidden:(2 * n_hidden), ...])
        return mean, std

    def log_prob(self, x, z):
        mean_std = self.net(z.view(-1, *z.size()[2:])).view(*z.size()[:2], x.size(1) * 2, *x.size()[3:])
        n_hidden = mean_std.size()[2] // 2
        mean = mean_std[:, :, :n_hidden, ...]
        var = torch.exp(mean_std[:, :, n_hidden:(2 * n_hidden), ...])
        log_p = - 0.5 * torch.prod(torch.tensor(z.size()[2:])) * np.log(2 * np.pi) \
                - 0.5 * torch.sum(torch.log(var) + (x.unsqueeze(1) - mean) ** 2 / var, list(range(2, z.dim())))
        return log_p


class NNBernoulliDecoder(Decoder):
    """
    Decoder representing a Bernoulli distribution with mean parametrized by a NN
    """

    def __init__(self, net):
        """
        Constructor
        :param net: neural network parametrizing mean Bernoulli (mean = sigmoid(nn_out)
        """
        super().__init__()
        self.net = net

    def forward(self, z):
        mean = torch.sigmoid(self.net(z))
        return mean

    def log_prob(self, x, z):
        score = self.net(z)
        x = x.unsqueeze(1)
        x = x.repeat(1, z.size()[0] // x.size()[0], *((x.dim() - 2) * [1])).view(-1, *x.size()[2:])
        log_sig = lambda a: -torch.relu(-a) - torch.log(1 + torch.exp(-torch.abs(a)))
        log_p = torch.sum(x * log_sig(score) + (1 - x) * log_sig(-score), list(range(1, x.dim())))
        return log_p



class PriorDistribution:
    def __init__(self):
        raise NotImplementedError

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        raise NotImplementedError


class ImagePrior(nn.Module):
    """
    Intensities of an image determine probability density of prior
    """
    def __init__(self, image, x_range=[-3, 3], y_range=[-3, 3], eps=1.e-10):
        """
        Constructor
        :param image: image as np matrix
        :param x_range: x range to position image at
        :param y_range: y range to position image at
        :param eps: small value to add to image to avoid log(0) problems
        """
        super().__init__()
        image_ = np.flip(image, 0).transpose() + eps
        self.image_cpu = torch.tensor(image_ / np.max(image_))
        self.image_size_cpu = self.image_cpu.size()
        self.x_range = torch.tensor(x_range)
        self.y_range = torch.tensor(y_range)

        self.register_buffer('image', self.image_cpu)
        self.register_buffer('image_size', torch.tensor(self.image_size_cpu).unsqueeze(0))
        self.register_buffer('density', torch.log(self.image_cpu / torch.sum(self.image_cpu)))
        self.register_buffer('scale', torch.tensor([[self.x_range[1] - self.x_range[0],
                                                     self.y_range[1] - self.y_range[0]]]))
        self.register_buffer('shift', torch.tensor([[self.x_range[0], self.y_range[0]]]))

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        z_ = torch.clamp((z - self.shift) / self.scale, max=1, min=0)
        ind = (z_ * (self.image_size - 1)).long()
        return self.density[ind[:, 0], ind[:, 1]]

    def rejection_sampling(self, num_steps=1):
        """
        Perform rejection sampling on image distribution
        :param num_steps: Number of rejection sampling steps to perform
        :return: Accepted samples
        """
        z_ = torch.rand((num_steps, 2), device=self.image.device)
        prob = torch.rand(num_steps, device=self.image.device)
        ind = (z_ * (self.image_size - 1)).long()
        intensity = self.image[ind[:, 0], ind[:, 1]]
        accept = intensity > prob
        z = z_[accept, :] * self.scale + self.shift
        return z

    def sample(self, num_samples=1):
        """
        Sample from image distribution through rejection sampling
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        z = torch.ones((0, 2), device=self.image.device)
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z


class TwoModes(PriorDistribution):
    def __init__(self, loc, scale):
        """
        Distribution 2d with two modes at z[0] = -loc and z[0] = loc
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.loc = loc
        self.scale = scale

    def log_prob(self, z):
        """
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        eps = torch.abs(torch.tensor(self.loc))

        log_prob = - 0.5 * ((torch.norm(z, dim=1) - self.loc) / (2 * self.scale)) ** 2\
                   - 0.5 * ((a - eps) / (3 * self.scale)) ** 2\
                   + torch.log(1 + torch.exp(-2 * (a * eps) / (3 * self.scale) ** 2))
        
        return log_prob
    
    
class Sinusoidal(PriorDistribution):
    def __init__(self, scale, period):
        """
        Distribution 2d with sinusoidal density
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.period = period

    def log_prob(self, z):
        """
        log(p) = - 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        w_1(z) = sin(2*pi / period * z[0])
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        w_1 = lambda x: torch.sin(2*np.pi / self.period * z_[0])
        log_prob = - 0.5 * ((z_[1] - w_1(z_)) / (self.scale)) ** 2 \
                    - 0.5 * (torch.norm(z_, dim=0, p=4) / (20 * self.scale)) ** 4 # add Gaussian envelope for valid p(z)
        
        return log_prob
    
    
class Sinusoidal_gap(PriorDistribution):
    def __init__(self, scale, period):
        """
        Distribution 2d with sinusoidal density with gap
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w2_scale = 0.6
        self.w2_amp = 3.0
        self.w2_mu = 1.0

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        w_1 = lambda x: torch.sin(2*np.pi / self.period * z_[0])
        w_2 = lambda x: self.w2_amp * torch.exp(-0.5*((z_[0] - self.w2_mu) / self.w2_scale)**2)

        eps = torch.abs(w_2(z_)/2)
        a = torch.abs(z_[1] - w_1(z_) + w_2(z_)/2)
        
        log_prob = -0.5 * ((a - eps) / self.scale) ** 2 + \
                    torch.log(1 + torch.exp(-2 * (eps * a) / self.scale**2)) \
                    - 0.5 * (torch.norm(z_, dim=0, p=4) / (20 * self.scale)) ** 4
        
        return log_prob
    
    
class Sinusoidal_split(PriorDistribution):
    def __init__(self, scale, period):
        """
        Distribution 2d with sinusoidal density with split
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w3_scale = 0.3
        self.w3_amp = 3.0
        self.w3_mu = 1.0

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        w_1 = lambda x: torch.sin(2*np.pi / self.period * z_[0])
        w_3 = lambda x: self.w3_amp * torch.sigmoid((z_[0] - self.w3_mu) / self.w3_scale)
        
        eps = torch.abs(w_3(z_)/2)
        a = torch.abs(z_[1] - w_1(z_) + w_3(z_)/2)
        
        log_prob = -0.5 * ((a - eps) / (self.scale))**2 + \
                    torch.log(1 + torch.exp(-2 * (eps * a) / self.scale**2)) \
                    - 0.5 * (torch.norm(z_, dim=0, p=4) / (20 * self.scale)) ** 4
        
        return log_prob
    
    
class Smiley(PriorDistribution):
    def __init__(self, scale):
        """
        Distribution 2d of a smiley :)
        :param loc: distance of modes from the origin
        :param scale: scale of modes
        """
        self.scale = scale
        self.loc = 2.0

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1, ) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z
            
        log_prob = - 0.5 * ((torch.norm(z_, dim=0) - self.loc) / (2 * self.scale)) ** 2\
                   - 0.5 * ((torch.abs(z_[1] + 0.8) - 1.2) / (2 * self.scale)) ** 2
        
        return log_prob



class MHProposal(nn.Module):
    """
    Proposal distribution for the Metropolis Hastings algorithm
    """
    def __init__(self):
        super().__init__()

    def sample(self, z):
        """
        Sample new value based on previous z
        """
        raise NotImplementedError

    def log_prob(self, z_, z):
        """
        :param z_: Potential new sample
        :param z: Previous sample
        :return: Log probability of proposal distribution
        """
        raise NotImplementedError

    def forward(self, z):
        """
        Draw samples given z and compute log probability difference
        log(p(z | z_new)) - log(p(z_new | z))
        :param z: Previous samples
        :return: Proposal, difference of log probability ratio
        """
        raise NotImplementedError


class DiagGaussianProposal(MHProposal):
    """
    Diagonal Gaussian distribution with previous value as mean
    as a proposal for Metropolis Hastings algorithm
    """
    def __init__(self, shape, scale):
        """
        Constructor
        :param shape: Shape of variables to sample
        :param scale: Standard deviation of distribution
        """
        super().__init__()
        self.shape = shape
        self.scale_cpu = torch.tensor(scale)
        self.register_buffer("scale", self.scale_cpu.unsqueeze(0))

    def sample(self, z):
        num_samples = len(z)
        eps = torch.randn((num_samples,) + self.shape, dtype=z.dtype, device=z.device)
        z_ = eps * self.scale + z
        return z_

    def log_prob(self, z_, z):
        log_p = - 0.5 * np.prod(self.shape) * np.log(2 * np.pi) \
                - torch.sum(torch.log(self.scale) + 0.5 * torch.pow((z_ - z) / self.scale, 2),
                            list(range(1, z.dim())))
        return log_p

    def forward(self, z):
        num_samples = len(z)
        eps = torch.randn((num_samples,) + self.shape, dtype=z.dtype, device=z.device)
        z_ = eps * self.scale + z
        log_p_diff = torch.zeros(num_samples, dtype=z.dtype, device=z.device)
        return z_, log_p_diff


class LinearInterpolation:
    """
    Linear interpolation of two distributions in the log space
    """
    def __init__(self, dist1, dist2, alpha):
        """
        Constructor
        :param dist1: First distribution
        :param dist2: Second distribution
        :param alpha: Interpolation parameter,
        log_p = alpha * log_p_1 + (1 - alpha) * log_p_2
        """
        self.alpha = alpha
        self.dist1 = dist1
        self.dist2 = dist2

    def log_prob(self, z):
        return self.alpha * self.dist1.log_prob(z)\
               + (1 - self.alpha) * self.dist2.log_prob(z)