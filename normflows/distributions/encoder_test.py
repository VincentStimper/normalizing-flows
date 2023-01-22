import unittest
import torch
import numpy as np

from torch.testing import assert_close

from normflows.nets import MLP
from normflows.distributions.encoder import Dirac, Uniform, \
    ConstDiagGaussian, NNDiagGaussian


class EncoderTest(unittest.TestCase):

    def checkForward(self, encoder, inputs, num_samples=1):
        # Do forward
        outputs, log_p = encoder(inputs, num_samples)
        # Check type
        assert log_p.dtype == inputs.dtype
        assert outputs.dtype == inputs.dtype
        # Check shape
        assert log_p.shape[0] == inputs.shape[0]
        assert outputs.shape[0] == inputs.shape[0]
        assert log_p.shape[1] == num_samples
        assert outputs.shape[1] == num_samples
        # Check dim
        assert outputs.dim() > log_p.dim()
        # Return results
        return outputs, log_p

    def checkLogProb(self, encoder, inputs_z, inputs_x):
        # Compute log prob
        log_p = encoder.log_prob(inputs_z, inputs_x)
        # Check type
        assert log_p.dtype == inputs_z.dtype
        # Check shape
        assert log_p.shape[0] == inputs_z.shape[0]
        # Return results
        return log_p

    def checkForwardLogProb(self, encoder, inputs, num_samples=1,
                            atol=None, rtol=None):
        # Check forward
        outputs, log_p = self.checkForward(encoder, inputs,
                                           num_samples)
        # Check log prob
        log_p_ = self.checkLogProb(encoder, outputs, inputs)
        # Check consistency
        assert_close(log_p_, log_p, atol=atol, rtol=rtol)

    def test_dirac_uniform(self):
        batch_size = 5
        encoder_class = [Dirac, Uniform]
        params = [(2, 1), (1, 3), (2, 2)]

        # Test model
        for n_dim, num_samples in params:
            for encoder_c in encoder_class:
                with self.subTest(n_dim=n_dim, num_samples=num_samples,
                                  encoder_c=encoder_c):
                    # Set up encoder
                    encoder = encoder_c()
                    # Do tests
                    inputs = torch.rand(batch_size, n_dim)
                    self.checkForwardLogProb(encoder, inputs, num_samples)

    def test_const_diag_gaussian(self):
        batch_size = 5
        params = [(2, 1), (1, 3), (2, 2)]

        # Test model
        for n_dim, num_samples in params:
            with self.subTest(n_dim=n_dim, num_samples=num_samples):
                # Set up encoder
                loc = np.random.randn(n_dim).astype(np.float32)
                scale = np.random.rand(n_dim).astype(np.float32) * 0.1 + 1
                encoder = ConstDiagGaussian(loc, scale)
                # Do tests
                inputs = torch.rand(batch_size, n_dim)
                self.checkForwardLogProb(encoder, inputs, num_samples)

    def test_nn_diag_gaussian(self):
        batch_size = 5
        n_hidden = 8
        params = [(4, 2, 1), (1, 1, 3), (5, 3, 2)]

        # Test model
        for n_dim, n_latent, num_samples in params:
            with self.subTest(n_dim=n_dim, n_latent=n_latent,
                              num_samples=num_samples):
                # Set up encoder
                nn = MLP([n_dim, n_hidden, n_latent * 2])
                encoder = NNDiagGaussian(nn)
                # Do tests
                inputs = torch.rand(batch_size, n_dim)
                self.checkForwardLogProb(encoder, inputs, num_samples)


if __name__ == "__main__":
    unittest.main()