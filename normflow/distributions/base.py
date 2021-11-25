import torch
import torch.nn as nn
import numpy as np

from .. import flows



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
    def __init__(self, shape, trainable=True):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1):
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype,
                          device=self.loc.device)
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - torch.sum(log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = - 0.5 * self.d * np.log(2 * np.pi)\
                - torch.sum(log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
                            list(range(1, self.n_dim + 1)))
        return log_p


class ClassCondDiagGaussian(BaseDistribution):
    """
    Class conditional multivariate Gaussian distribution with diagonal covariance matrix
    """
    def __init__(self, shape, num_classes):
        """
        Constructor
        :param shape: Tuple with shape of data, if int shape has one dimension
        :param num_classes: Number of classes to condition on
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.perm = [self.n_dim] + list(range(self.n_dim))
        self.d = np.prod(shape)
        self.num_classes = num_classes
        self.loc = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.log_scale = nn.Parameter(torch.zeros(*self.shape, num_classes))
        self.temperature = None # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, y=None):
        if y is not None:
            num_samples = len(y)
        else:
            y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
        if y.dim() == 1:
            y_onehot = torch.zeros((self.num_classes, num_samples), dtype=self.loc.dtype,
                                   device=self.loc.device)
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype,
                          device=self.loc.device)
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        z = loc + torch.exp(log_scale) * eps
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - torch.sum(log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z, y):
        if y.dim() == 1:
            y_onehot = torch.zeros((self.num_classes, len(y)), dtype=self.loc.dtype,
                                   device=self.loc.device)
            y_onehot.scatter_(0, y[None], 1)
            y = y_onehot
        else:
            y = y.t()
        loc = (self.loc @ y).permute(*self.perm)
        log_scale = (self.log_scale @ y).permute(*self.perm)
        if self.temperature is not None:
            log_scale = np.log(self.temperature) + log_scale
        log_p = - 0.5 * self.d * np.log(2 * np.pi)\
                - torch.sum(log_scale + 0.5 * torch.pow((z - loc) / torch.exp(log_scale), 2),
                            list(range(1, self.n_dim + 1)))
        return log_p


class GlowBase(BaseDistribution):
    """
    Base distribution of the Glow model, i.e. Diagonal Gaussian with one mean and
    log scale for each channel
    """
    def __init__(self, shape, num_classes=None, logscale_factor=3.):
        """
        Constructor
        :param shape: Shape of the variables
        :param num_classes: Number of classes if the base is class conditional,
        None otherwise
        :param logscale_factor: Scaling factor for mean and log variance
        """
        super().__init__()
        # Save shape and related statistics
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_dim = len(shape)
        self.num_pix = np.prod(shape[1:])
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        self.logscale_factor = logscale_factor
        # Set up parameters
        self.loc = nn.Parameter(torch.zeros(1, self.shape[0], *((self.n_dim - 1) * [1])))
        self.loc_logs = nn.Parameter(torch.zeros(1, self.shape[0],
                                                 *((self.n_dim - 1) * [1])))
        self.log_scale = nn.Parameter(torch.zeros(1, self.shape[0],
                                                  *((self.n_dim - 1) * [1])))
        self.log_scale_logs = nn.Parameter(torch.zeros(1, self.shape[0],
                                                       *((self.n_dim - 1) * [1])))
        # Class conditional parameter if needed
        if self.class_cond:
            self.loc_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
            self.log_scale_cc = nn.Parameter(torch.zeros(self.num_classes, self.shape[0]))
        # Temperature parameter for annealed sampling
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        # Prepare parameter
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(self.log_scale_logs * self.logscale_factor)
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=self.loc.device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype,
                                       device=self.loc.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(y.size(0), self.shape[0],
                                               *((self.n_dim - 1) * [1]))
            log_scale = log_scale + (y @ self.log_scale_cc).view(y.size(0), self.shape[0],
                                                                 *((self.n_dim - 1) * [1]))
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        # Sample
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype,
                          device=self.loc.device)
        z = loc + torch.exp(log_scale) * eps
        # Get log prob
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - self.num_pix * torch.sum(log_scale, dim=self.sum_dim) \
                - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        return z, log_p

    def log_prob(self, z, y=None):
        # Perpare parameter
        loc = self.loc * torch.exp(self.loc_logs * self.logscale_factor)
        log_scale = self.log_scale * torch.exp(self.log_scale_logs * self.logscale_factor)
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=self.loc.dtype,
                                       device=self.loc.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
            loc = loc + (y @ self.loc_cc).view(y.size(0), self.shape[0],
                                               *((self.n_dim - 1) * [1]))
            log_scale = log_scale + (y @ self.log_scale_cc).view(y.size(0), self.shape[0],
                                                                 *((self.n_dim - 1) * [1]))
        if self.temperature is not None:
            log_scale = log_scale + np.log(self.temperature)
        # Get log prob
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - self.num_pix * torch.sum(log_scale, dim=self.sum_dim)\
                - 0.5 * torch.sum(torch.pow((z - loc) / torch.exp(log_scale), 2),
                                  dim=self.sum_dim)
        return log_p


class AffineGaussian(BaseDistribution):
    """
    Diagonal Gaussian an affine constant transformation applied to it,
    can be class conditional or not
    """
    def __init__(self, shape, affine_shape, num_classes=None):
        """
        Constructor
        :param shape: Shape of the variables
        :param affine_shape: Shape of the parameters in the affine transformation
        :param num_classes: Number of classes if the base is class conditional,
        None otherwise
        """
        super().__init__()
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.affine_shape = affine_shape
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        # Affine transformation
        if self.class_cond:
            self.transform = flows.CCAffineConst(self.affine_shape, self.num_classes)
        else:
            self.transform = flows.AffineConstFlow(self.affine_shape)
        # Temperature parameter for annealed sampling
        self.temperature = None

    def forward(self, num_samples=1, y=None):
        dtype = self.transform.s.dtype
        device = self.transform.s.device
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=dtype, device=device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.
        # Sample
        eps = torch.randn((num_samples,) + self.shape, dtype=dtype, device=device)
        z = np.exp(log_scale) * eps
        # Get log prob
        log_p = - 0.5 * self.d * np.log(2 * np.pi) \
                - self.d * log_scale \
                - 0.5 * torch.sum(torch.pow(eps, 2), dim=self.sum_dim)
        # Apply transform
        if self.class_cond:
            z, log_det = self.transform(z, y)
        else:
            z, log_det = self.transform(z)
        log_p -= log_det
        return z, log_p

    def log_prob(self, z, y=None):
        # Perpare onehot encoding of class if needed
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes),
                                       dtype=self.transform.s.dtype,
                                       device=self.transform.s.device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        if self.temperature is not None:
            log_scale = np.log(self.temperature)
        else:
            log_scale = 0.
        # Get log prob
        if self.class_cond:
            z, log_p = self.transform.inverse(z, y)
        else:
            z, log_p = self.transform.inverse(z)
        z = z / np.exp(log_scale)
        log_p = log_p - self.d * log_scale \
                - 0.5 * self.d * np.log(2 * np.pi) \
                - 0.5 * torch.sum(torch.pow(z, 2), dim=self.sum_dim)
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