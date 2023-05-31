import unittest

from normflows.distributions.base import DiagGaussian
from normflows.distributions.target import TwoMoons, \
    CircularGaussianMixture, RingMixture, TwoIndependent


class TargetTest(unittest.TestCase):
    def test_targets(self):
        targets = [TwoMoons, CircularGaussianMixture,
                   RingMixture]
        for num_samples in [1, 5]:
            for target_ in targets:
                with self.subTest(num_samples=num_samples,
                                  target_=target_):
                    # Set up target
                    target = target_()
                    # Test target
                    samples = target.sample(num_samples)
                    assert samples.shape == (num_samples, 2)
                    log_p = target.log_prob(samples)
                    assert log_p.shape == (num_samples,)

    def test_two_independent(self):
        target = TwoIndependent(TwoMoons(), DiagGaussian(2))
        for num_samples in [1, 5]:
            with self.subTest(num_samples=num_samples):
                # Test target
                samples = target.sample(num_samples)
                assert samples.shape == (num_samples, 4)
                log_p = target.log_prob(samples)
                assert log_p.shape == (num_samples,)


if __name__ == "__main__":
    unittest.main()