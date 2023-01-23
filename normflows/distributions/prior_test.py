import unittest
import torch
import numpy as np

from normflows.distributions.prior import ImagePrior, TwoModes, \
    Sinusoidal, Sinusoidal_split, Sinusoidal_gap, Smiley


class PriorTest(unittest.TestCase):
    def test_2d_priors(self):
        batch_size = 5
        prior_params = [(TwoModes, {'loc': 1, 'scale': 0.2}),
                        (Sinusoidal, {'scale': 1., 'period': 1.}),
                        (Sinusoidal_split, {'scale': 1., 'period': 1.}),
                        (Sinusoidal_gap, {'scale': 1., 'period': 1.}),
                        (Smiley, {'scale': 1.})]

        # Test model
        for prior_c, params in prior_params:
            with self.subTest(prior_c=prior_c, params=params):
                # Set up prior
                prior = prior_c(**params)
                # Test prior
                inputs = torch.randn(batch_size, 2)
                log_p = prior.log_prob(inputs)
                assert log_p.shape == (batch_size,)
                assert log_p.dtype == inputs.dtype

    def test_image_prior(self):
        # Set up prior
        image = np.random.rand(10, 10).astype(np.float32)
        prior = ImagePrior(image)
        for num_samples in [1, 5]:
            with self.subTest(num_samples=num_samples):
                # Test prior
                samples = prior.sample(num_samples)
                assert samples.shape == (num_samples, 2)
                assert samples.dtype == torch.float32
                log_p = prior.log_prob(samples)
                assert log_p.shape == (num_samples,)
                assert log_p.dtype == torch.float32


if __name__ == "__main__":
    unittest.main()