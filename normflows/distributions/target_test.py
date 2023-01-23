import unittest
import torch

from normflows.distributions.target import TwoMoons, \
    CircularGaussianMixture, RingMixture


class TargetTest(unittest.TestCase):
    def test_targets(self):
        targets = [TwoMoons, CircularGaussianMixture,
                   RingMixture]
        for num_samples in [1, 5]:
            for target_ in targets:
                with self.subTest(num_samples=num_samples,
                                  target_=target_):
                    # Set up prior
                    target = target_()
                    # Test prior
                    samples = target.sample(num_samples)
                    assert samples.shape == (num_samples, 2)
                    log_p = target.log_prob(samples)
                    assert log_p.shape == (num_samples,)


if __name__ == "__main__":
    unittest.main()