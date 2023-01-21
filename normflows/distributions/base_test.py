import unittest
import torch
import numpy as np

from normflows.distributions.base import DiagGaussian, UniformGaussian
from normflows.distributions.distribution_test import DistributionTest


class BaseTest(DistributionTest):
    def test_diag_gaussian(self):
        for shape in [1, (3,), [2, 3]]:
            for num_samples in [1, 3]:
                with self.subTest(shape=shape, num_samples=num_samples):
                    distribution = DiagGaussian(shape)
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


if __name__ == "__main__":
    unittest.main()