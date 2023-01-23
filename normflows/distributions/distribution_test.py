import unittest

from torch.testing import assert_close


class DistributionTest(unittest.TestCase):
    """
    Generic test case for distribution modules
    """
    def assertClose(self, actual, expected, atol=None, rtol=None):
        assert_close(actual, expected, atol=atol, rtol=rtol)

    def checkForward(self, distribution, num_samples=1, **kwargs):
        # Do forward
        outputs, log_p = distribution(num_samples, **kwargs)
        # Check type
        assert outputs.dtype == log_p.dtype
        # Check shape
        assert log_p.shape[0] == num_samples
        assert outputs.shape[0] == num_samples
        # Check dim
        assert outputs.dim() > log_p.dim()
        # Return results
        return outputs, log_p

    def checkLogProb(self, distribution, inputs, **kwargs):
        # Compute log prob
        log_p = distribution.log_prob(inputs, **kwargs)
        # Check type
        assert log_p.dtype == inputs.dtype
        # Check shape
        assert log_p.shape[0] == inputs.shape[0]
        # Return results
        return log_p

    def checkSample(self, distribution, num_samples=1, **kwargs):
        # Do forward
        outputs = distribution.sample(num_samples, **kwargs)
        # Check shape
        assert outputs.shape[0] == num_samples
        # Check dim
        assert outputs.dim() > 1
        # Return results
        return outputs

    def checkForwardLogProb(self, distribution, num_samples=1, atol=None, rtol=None, **kwargs):
        # Check forward
        outputs, log_p = self.checkForward(distribution, num_samples, **kwargs)
        # Check log prob
        log_p_ = self.checkLogProb(distribution, outputs, **kwargs)
        # Check consistency
        self.assertClose(log_p_, log_p, atol, rtol)