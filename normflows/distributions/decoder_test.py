import unittest
import torch

from normflows.nets import MLP
from normflows.distributions.decoder import NNDiagGaussianDecoder, \
    NNBernoulliDecoder


class DecoderTest(unittest.TestCase):

    def test_decoder(self):
        batch_size = 5
        n_dim = 10
        n_bottleneck = 3
        n_hidden_untis = 16
        hidden_units_decoder = [n_bottleneck, n_hidden_untis]
        decoder_class = [NNDiagGaussianDecoder, NNBernoulliDecoder]
        n_parameter = [2, 1]

        # Test model
        for num_samples in [1, 4]:
            for i in range(len(decoder_class)):
                with self.subTest(num_samples=num_samples, i=i):
                    # Set up decoder
                    decoder_nn = MLP(hidden_units_decoder + [n_parameter[i] * n_dim])
                    decoder = decoder_class[i](decoder_nn)

                    # Test decoder
                    x = torch.randn((batch_size, n_dim))
                    z = torch.randn((batch_size * num_samples, n_bottleneck))
                    if n_parameter[i] == 1:
                        para = decoder(z)
                        assert para.shape == (batch_size * num_samples, n_dim)
                    elif n_parameter[i] == 2:
                        para1, para2 = decoder(z)
                        assert para1.shape == (batch_size * num_samples, n_dim)
                        assert para2.shape == (batch_size * num_samples, n_dim)
                    log_p = decoder.log_prob(x, z)
                    assert log_p.shape == (batch_size * num_samples,)


if __name__ == "__main__":
    unittest.main()