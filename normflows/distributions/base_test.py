import unittest
import torch
import numpy as np

from normflows.distributions.base import DiagGaussian, UniformGaussian, \
    ClassCondDiagGaussian, GlowBase, AffineGaussian, GaussianMixture, \
    GaussianPCA, Uniform, ConditionalDiagGaussian
from normflows.distributions.distribution_test import DistributionTest
from normflows.nets.mlp import MLP


class BaseTest(DistributionTest):
    def test_diag_gaussian(self):
        for shape in [1, (3,), [2, 3]]:
            for num_samples in [1, 3]:
                with self.subTest(shape=shape, num_samples=num_samples):
                    distribution = DiagGaussian(shape)
                    self.checkForwardLogProb(distribution, num_samples)
                    _ = self.checkSample(distribution, num_samples)

    def test_conditional_diag_gaussian(self):
        context_size = 5
        hidden_units = 16
        for shape, base_param_size in [(1, 2), ([3], 6)]:
            for num_samples in [1, 3]:
                with self.subTest(shape=shape, base_param_size=base_param_size,
                                  num_samples=num_samples):
                    context_encoder = MLP([context_size, hidden_units,
                                           base_param_size])
                    distribution = ConditionalDiagGaussian(shape, context_encoder)
                    context = torch.randn(num_samples, context_size)
                    self.checkForwardLogProb(distribution, num_samples,
                                             context=context)
                    _ = self.checkSample(distribution, num_samples,
                                         context=context)

    def test_uniform(self):
        for shape in [1, (3,), [2, 3]]:
            for num_samples in [1, 3]:
                with self.subTest(shape=shape, num_samples=num_samples):
                    distribution = Uniform(shape)
                    self.checkForwardLogProb(distribution, num_samples)
                    _ = self.checkSample(distribution, num_samples)

    def test_uniform_gaussian(self):
        params = [(2, 1, None), (2, (0,), 0.5 * torch.ones(2)),
                  (4, [2], None), (3, [2, 0], np.pi * torch.ones(3))]
        for ndim, ind, scale in params:
            for num_samples in [1, 3]:
                with self.subTest(ndim=ndim, ind=ind, scale=scale,
                                  num_samples=num_samples):
                    distribution = UniformGaussian(ndim, ind, scale)
                    self.checkForwardLogProb(distribution, num_samples)
                    _ = self.checkSample(distribution, num_samples)

    def test_cc_diag_gaussian(self):
        params = [(1, 3), ((3,), 1), ([2, 3], 5)]
        for shape, num_classes in params:
            for num_samples in [1, 3]:
                with self.subTest(shape=shape, num_classes=num_classes,
                                  num_samples=num_samples):
                    distribution = ClassCondDiagGaussian(shape, num_classes)
                    y = torch.randint(num_classes, (num_samples,))
                    self.checkForwardLogProb(distribution, num_samples, y=y)
                    _ = self.checkSample(distribution, num_samples)

    def test_glow_base(self):
        params = [(1, 3), ((3,), 1), ([2, 3], None), ((3, 2, 2), 5)]
        for shape, num_classes in params:
            for num_samples in [1, 3]:
                with self.subTest(shape=shape, num_classes=num_classes,
                                  num_samples=num_samples):
                    distribution = GlowBase(shape, num_classes)
                    if num_classes is None:
                        y = None
                    else:
                        y = torch.randint(num_classes, (num_samples,))
                    self.checkForwardLogProb(distribution, num_samples, y=y)
                    _ = self.checkSample(distribution, num_samples)

    def test_affine_gaussian(self):
        params = [(1, (1,), 2), ((3,), 1, 1),
                  ([2, 3], (2, 1), None), ((3, 2, 2), (3, 1, 2), 5)]
        for shape, affine_shape, num_classes in params:
            for num_samples in [1, 3]:
                with self.subTest(shape=shape, num_classes=num_classes,
                                  num_samples=num_samples):
                    distribution = AffineGaussian(shape, affine_shape,
                                                  num_classes)
                    if num_classes is None:
                        y = None
                    else:
                        y = torch.randint(num_classes, (num_samples,))
                    self.checkForwardLogProb(distribution, num_samples, y=y)
                    _ = self.checkSample(distribution, num_samples)

    def test_gaussian_mixture(self):
        params = [(4, 1, True), (1, 2, False), (2, 3, True)]
        for n_modes, dim, trainable in params:
            for num_samples in [1, 3]:
                with self.subTest(n_modes=n_modes, dim=dim,
                                  trainable=trainable,
                                  num_samples=num_samples):
                    distribution = GaussianMixture(n_modes, dim,
                                                   trainable)
                    self.checkForwardLogProb(distribution, num_samples)
                    _ = self.checkSample(distribution, num_samples)

    def test_gaussian_pca(self):
        params = [(1, 1), (4, 2), (5, 1)]
        for dim, latent_dim in params:
            for num_samples in [1, 3]:
                with self.subTest(dim=dim, latent_dim=latent_dim):
                    distribution = GaussianPCA(dim, latent_dim)
                    self.checkForwardLogProb(distribution, num_samples)
                    _ = self.checkSample(distribution, num_samples)


if __name__ == "__main__":
    unittest.main()