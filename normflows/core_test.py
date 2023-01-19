import unittest
import torch

from torch.testing import assert_close
from normflows import NormalizingFlow
from normflows.flows import MaskedAffineFlow
from normflows.nets import MLP
from normflows.distributions.base import DiagGaussian
from normflows.distributions.target import CircularGaussianMixture


class CoreTest(unittest.TestCase):
    def test_mask_affine(self):
        batch_size = 5
        latent_size = 2
        for n_layers in [2, 5]:
            with self.subTest(n_layers=n_layers):
                # Construct flow model
                layers = []
                for i in range(n_layers):
                    b = torch.Tensor([j if i % 2 == j % 2 else 0 for j in range(latent_size)])
                    s = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                    t = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                    layers.append(MaskedAffineFlow(b, t, s))
                base = DiagGaussian(latent_size)
                target = CircularGaussianMixture()
                model = NormalizingFlow(base, layers, target)
                inputs = torch.randn((batch_size, latent_size))
                # Test log prob and sampling
                log_q = model.log_prob(inputs)
                assert log_q.shape == (batch_size,)
                s, log_q = model.sample(batch_size)
                assert log_q.shape == (batch_size,)
                assert s.shape == (batch_size, latent_size)
                # Test losses
                loss = model.forward_kld(inputs)
                assert loss.dim() == 0
                loss = model.reverse_kld(batch_size)
                assert loss.dim() == 0
                loss = model.reverse_alpha_div(batch_size)
                assert loss.dim() == 0
                # Test forward and inverse
                outputs = model.forward(inputs)
                inputs_ = model.inverse(outputs)
                assert_close(inputs_, inputs)
                outputs, log_det = model.forward_and_log_det(inputs)
                inputs_, log_det_ = model.inverse_and_log_det(outputs)
                assert_close(inputs_, inputs)
                assert_close(log_det, -log_det_)


if __name__ == "__main__":
    unittest.main()