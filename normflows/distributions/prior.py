import numpy as np
import torch
from torch import nn


class PriorDistribution:
    def __init__(self):
        raise NotImplementedError

    def log_prob(self, z):
        """
        Args:
         z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        raise NotImplementedError


class ImagePrior(nn.Module):
    """
    Intensities of an image determine probability density of prior
    """

    def __init__(self, image, x_range=[-3, 3], y_range=[-3, 3], eps=1.0e-10):
        """Constructor

        Args:
          image: image as np matrix
          x_range: x range to position image at
          y_range: y range to position image at
          eps: small value to add to image to avoid log(0) problems
        """
        super().__init__()
        image_ = np.flip(image, 0).transpose() + eps
        self.image_cpu = torch.tensor(image_ / np.max(image_))
        self.image_size_cpu = self.image_cpu.size()
        self.x_range = torch.tensor(x_range)
        self.y_range = torch.tensor(y_range)

        self.register_buffer("image", self.image_cpu)
        self.register_buffer(
            "image_size", torch.tensor(self.image_size_cpu).unsqueeze(0)
        )
        self.register_buffer(
            "density", torch.log(self.image_cpu / torch.sum(self.image_cpu))
        )
        self.register_buffer(
            "scale",
            torch.tensor(
                [[self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]]]
            ),
        )
        self.register_buffer(
            "shift", torch.tensor([[self.x_range[0], self.y_range[0]]])
        )

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        z_ = torch.clamp((z - self.shift) / self.scale, max=1, min=0)
        ind = (z_ * (self.image_size - 1)).long()
        return self.density[ind[:, 0], ind[:, 1]]

    def rejection_sampling(self, num_steps=1):
        """Perform rejection sampling on image distribution

        Args:
         num_steps: Number of rejection sampling steps to perform

        Returns:
          Accepted samples
        """
        z_ = torch.rand(
            (num_steps, 2), dtype=self.image.dtype, device=self.image.device
        )
        prob = torch.rand(num_steps, dtype=self.image.dtype, device=self.image.device)
        ind = (z_ * (self.image_size - 1)).long()
        intensity = self.image[ind[:, 0], ind[:, 1]]
        accept = intensity > prob
        z = z_[accept, :] * self.scale + self.shift
        return z

    def sample(self, num_samples=1):
        """Sample from image distribution through rejection sampling

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples
        """
        z = torch.ones((0, 2), dtype=self.image.dtype, device=self.image.device)
        while len(z) < num_samples:
            z_ = self.rejection_sampling(num_samples)
            ind = np.min([len(z_), num_samples - len(z)])
            z = torch.cat([z, z_[:ind, :]], 0)
        return z


class TwoModes(PriorDistribution):
    def __init__(self, loc, scale):
        """Distribution 2d with two modes

        Distribution 2d with two modes at
        ```z[0] = -loc```  and ```z[0] = loc```
        following the density
        ```
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        ```

        Args:
          loc: distance of modes from the origin
          scale: scale of modes
        """
        self.loc = loc
        self.scale = scale

    def log_prob(self, z):
        """

        ```
        log(p) = 1/2 * ((norm(z) - loc) / (2 * scale)) ** 2
                - log(exp(-1/2 * ((z[0] - loc) / (3 * scale)) ** 2) + exp(-1/2 * ((z[0] + loc) / (3 * scale)) ** 2))
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        a = torch.abs(z[:, 0])
        eps = torch.abs(torch.tensor(self.loc))

        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - self.loc) / (2 * self.scale)) ** 2
            - 0.5 * ((a - eps) / (3 * self.scale)) ** 2
            + torch.log(1 + torch.exp(-2 * (a * eps) / (3 * self.scale) ** 2))
        )

        return log_prob


class Sinusoidal(PriorDistribution):
    def __init__(self, scale, period):
        """Distribution 2d with sinusoidal density
        given by

        ```
        w_1(z) = sin(2*pi / period * z[0])
        log(p) = - 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        ```

        Args:
          scale: scale of the distribution, see formula
          period: period of the sinosoidal
        """
        self.scale = scale
        self.period = period

    def log_prob(self, z):
        """

        ```
        log(p) = - 1/2 * ((z[1] - w_1(z)) / (2 * scale)) ** 2
        w_1(z) = sin(2*pi / period * z[0])
        ```

        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1,) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z

        w_1 = lambda x: torch.sin(2 * np.pi / self.period * z_[0])
        log_prob = (
            -0.5 * ((z_[1] - w_1(z_)) / (self.scale)) ** 2
            - 0.5 * (torch.norm(z_, dim=0, p=4) / (20 * self.scale)) ** 4
        )  # add Gaussian envelope for valid p(z)

        return log_prob


class Sinusoidal_gap(PriorDistribution):
    def __init__(self, scale, period):
        """Distribution 2d with sinusoidal density with gap
        given by

        ```
        w_1(z) = sin(2*pi / period * z[0])
        w_2(z) = 3 * exp(-0.5 * ((z[0] - 1) / 0.6) ** 2)
        log(p) = -log(exp(-0.5 * ((z[1] - w_1(z)) / 0.35) ** 2) + exp(-0.5 * ((z[1] - w_1(z) + w_2(z)) / 0.35) ** 2))
        ```

        Args:
          loc: distance of modes from the origin
          scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w2_scale = 0.6
        self.w2_amp = 3.0
        self.w2_mu = 1.0

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1,) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z

        w_1 = lambda x: torch.sin(2 * np.pi / self.period * z_[0])
        w_2 = lambda x: self.w2_amp * torch.exp(
            -0.5 * ((z_[0] - self.w2_mu) / self.w2_scale) ** 2
        )

        eps = torch.abs(w_2(z_) / 2)
        a = torch.abs(z_[1] - w_1(z_) + w_2(z_) / 2)

        log_prob = (
            -0.5 * ((a - eps) / self.scale) ** 2
            + torch.log(1 + torch.exp(-2 * (eps * a) / self.scale**2))
            - 0.5 * (torch.norm(z_, dim=0, p=4) / (20 * self.scale)) ** 4
        )

        return log_prob


class Sinusoidal_split(PriorDistribution):
    def __init__(self, scale, period):
        """Distribution 2d with sinusoidal density with split
        given by

        ```
        w_1(z) = sin(2*pi / period * z[0])
        w_3(z) = 3 * sigmoid((z[0] - 1) / 0.3)
        log(p) = -log(exp(-0.5 * ((z[1] - w_1(z)) / 0.4) ** 2) + exp(-0.5 * ((z[1] - w_1(z) + w_3(z)) / 0.35) ** 2))
        ```

        Args:
          loc: distance of modes from the origin
          scale: scale of modes
        """
        self.scale = scale
        self.period = period
        self.w3_scale = 0.3
        self.w3_amp = 3.0
        self.w3_mu = 1.0

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1,) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z

        w_1 = lambda x: torch.sin(2 * np.pi / self.period * z_[0])
        w_3 = lambda x: self.w3_amp * torch.sigmoid(
            (z_[0] - self.w3_mu) / self.w3_scale
        )

        eps = torch.abs(w_3(z_) / 2)
        a = torch.abs(z_[1] - w_1(z_) + w_3(z_) / 2)

        log_prob = (
            -0.5 * ((a - eps) / (self.scale)) ** 2
            + torch.log(1 + torch.exp(-2 * (eps * a) / self.scale**2))
            - 0.5 * (torch.norm(z_, dim=0, p=4) / (20 * self.scale)) ** 4
        )

        return log_prob


class Smiley(PriorDistribution):
    def __init__(self, scale):
        """Distribution 2d of a smiley :)

        Args:
          scale: scale of the smiley
        """
        self.scale = scale
        self.loc = 2.0

    def log_prob(self, z):
        """
        Args:
          z: value or batch of latent variable

        Returns:
          log probability of the distribution for z
        """
        if z.dim() > 1:
            z_ = z.permute((z.dim() - 1,) + tuple(range(0, z.dim() - 1)))
        else:
            z_ = z

        log_prob = (
            -0.5 * ((torch.norm(z_, dim=0) - self.loc) / (2 * self.scale)) ** 2
            - 0.5 * ((torch.abs(z_[1] + 0.8) - 1.2) / (2 * self.scale)) ** 2
        )

        return log_prob
