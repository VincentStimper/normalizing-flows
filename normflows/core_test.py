import unittest
import torch

from torch.testing import assert_close
from normflows import NormalizingFlow, ClassCondFlow, \
    MultiscaleFlow, NormalizingFlowVAE, \
    ConditionalNormalizingFlow
from normflows.flows import MaskedAffineFlow, GlowBlock, \
    Merge, Squeeze, MaskedAffineAutoregressive, \
    AutoregressiveRationalQuadraticSpline
from normflows.nets import MLP
from normflows.distributions.base import DiagGaussian, \
    ClassCondDiagGaussian
from normflows.distributions.target import CircularGaussianMixture, \
    ConditionalDiagGaussian
from normflows.distributions.encoder import NNDiagGaussian
from normflows.distributions.decoder import NNDiagGaussianDecoder


class CoreTest(unittest.TestCase):
    def test_normalizing_flow(self):
        batch_size = 5
        latent_size = 2
        for n_layers in [2, 5]:
            with self.subTest(n_layers=n_layers):
                # Construct flow model
                layers = []
                for i in range(n_layers):
                    b = torch.Tensor([j if i % 2 == j % 2 else 0 for j in range(latent_size)])
                    s = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=False)
                    t = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=False)
                    layers.append(MaskedAffineFlow(b, t, s))
                base = DiagGaussian(latent_size)
                target = CircularGaussianMixture()
                model = NormalizingFlow(base, layers, target)
                # Test log prob and sampling
                inputs = torch.randn((batch_size, latent_size))
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
                loss = model.reverse_kld(batch_size, score_fn=False)
                assert loss.dim() == 0
                loss = model.reverse_alpha_div(batch_size)
                assert loss.dim() == 0
                loss = model.reverse_alpha_div(batch_size, dreg=True)
                assert loss.dim() == 0
                # Test forward and inverse
                outputs = model.forward(inputs)
                inputs_ = model.inverse(outputs)
                assert_close(inputs_, inputs)
                outputs, log_det = model.forward_and_log_det(inputs)
                inputs_, log_det_ = model.inverse_and_log_det(outputs)
                assert_close(inputs_, inputs)
                assert_close(log_det, -log_det_)

    def test_conditional_normalizing_flow(self):
        batch_size = 5
        latent_size = 2
        context_size = latent_size * 2
        n_layers = 2
        for flow_layer_type in ['maf', 'auto_rqs']:
            with self.subTest(n_layers=n_layers):
                # Construct flow model
                layers = []
                for i in range(n_layers):
                    if flow_layer_type == 'maf':
                        layer = MaskedAffineAutoregressive(
                            features=latent_size,
                            hidden_features=32,
                            context_features=context_size,
                            num_blocks=2,
                            use_residual_blocks=True,
                            random_mask=False,
                        )
                    elif flow_layer_type == 'auto_rqs':
                        layer = AutoregressiveRationalQuadraticSpline(
                            num_input_channels=latent_size,
                            num_blocks=2,
                            num_hidden_channels=32,
                            num_context_channels=context_size,
                        )
                    layers.append(layer)
                base = DiagGaussian(latent_size)
                target = ConditionalDiagGaussian()
                model = ConditionalNormalizingFlow(base, layers, target)
                # Test log prob and sampling
                inputs = torch.randn((batch_size, latent_size))
                context = torch.rand((batch_size, context_size)) + 0.5
                log_q = model.log_prob(inputs, context)
                assert log_q.shape == (batch_size,)
                s, log_q = model.sample(batch_size, context)
                assert log_q.shape == (batch_size,)
                assert s.shape == (batch_size, latent_size)
                # Test losses
                loss = model.forward_kld(inputs, context)
                assert loss.dim() == 0
                loss = model.reverse_kld(batch_size, context)
                assert loss.dim() == 0
                loss = model.reverse_kld(batch_size, context, score_fn=False)
                assert loss.dim() == 0
                # Test forward and inverse
                outputs = model.forward(inputs, context)
                inputs_ = model.inverse(outputs, context)
                assert_close(inputs_, inputs)
                outputs, log_det = model.forward_and_log_det(inputs, context)
                inputs_, log_det_ = model.inverse_and_log_det(outputs, context)
                assert_close(inputs_, inputs)
                assert_close(log_det, -log_det_)

    def test_cc_normalizing_flow(self):
        batch_size = 5
        latent_size = 2
        n_layers = 2
        n_classes = 3

        # Construct flow model
        layers = []
        for i in range(n_layers):
            b = torch.Tensor([j if i % 2 == j % 2 else 0 for j in range(latent_size)])
            s = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
            t = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
            layers.append(MaskedAffineFlow(b, t, s))
        base = ClassCondDiagGaussian(latent_size, n_classes)
        model = ClassCondFlow(base, layers)

        # Test model
        x = torch.randn((batch_size, latent_size))
        y = torch.randint(n_classes, (batch_size,))
        log_q = model.log_prob(x, y)
        assert log_q.shape == (batch_size,)
        s, log_q = model.sample(x, y)
        assert log_q.shape == (batch_size,)
        assert s.shape == (batch_size, latent_size)
        loss = model.forward_kld(x, y)
        assert loss.dim() == 0

    def test_multiscale_flow(self):
        # Set parameters
        batch_size = 5
        K = 2
        hidden_channels = 32
        split_mode = 'channel'
        scale = True
        params = [(2, 1, (3, 8, 8)), (3, 5, (1, 16, 16))]

        for L, num_classes, input_shape in params:
            with self.subTest(L=L, num_classes=num_classes,
                              input_shape=input_shape):
                # Set up flows, distributions and merge operations
                base = []
                merges = []
                flows = []
                for i in range(L):
                    flows_ = []
                    for j in range(K):
                        flows_ += [GlowBlock(input_shape[0] * 2 ** (L + 1 - i), hidden_channels,
                                             split_mode=split_mode, scale=scale)]
                    flows_ += [Squeeze()]
                    flows += [flows_]
                    if i > 0:
                        merges += [Merge()]
                        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                                        input_shape[2] // 2 ** (L - i))
                    else:
                        latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                                        input_shape[2] // 2 ** L)
                    base += [ClassCondDiagGaussian(latent_shape, num_classes)]

                # Construct flow model
                model = MultiscaleFlow(base, flows, merges)
                # Test model
                y = torch.randint(num_classes, (batch_size,))
                x, log_q = model.sample(batch_size, y)
                log_q_ = model.log_prob(x, y)
                assert x.shape == (batch_size,) + (input_shape)
                assert log_q.shape == (batch_size,)
                assert log_q_.shape == (batch_size,)
                assert log_q.dtype == x.dtype
                assert log_q_.dtype == x.dtype
                assert_close(log_q, log_q_)
                fwd = model.forward(x, y)
                fwd_kld = model.forward_kld(x, y)
                assert_close(torch.mean(fwd), fwd_kld)
                z, log_det = model.inverse_and_log_det(x)
                x_, log_det_ = model.forward_and_log_det(z)
                assert len(z) == L
                assert x_.shape == (batch_size,) + (input_shape)
                assert_close(x_, x)
                assert_close(log_det, -log_det_)


    def test_normalizing_flow_vae(self):
        batch_size = 5
        n_dim = 10
        n_layers = 2
        n_bottleneck = 3
        n_hidden_untis = 16
        hidden_units_encoder = [n_dim, n_hidden_untis, n_bottleneck * 2]
        hidden_units_decoder = [n_bottleneck, n_hidden_untis, 2 * n_dim]

        # Construct flow model
        layers = []
        for i in range(n_layers):
            b = torch.Tensor([j if i % 2 == j % 2 else 0 for j in range(n_bottleneck)])
            s = MLP([n_bottleneck, 2 * n_bottleneck, n_bottleneck], init_zeros=True)
            t = MLP([n_bottleneck, 2 * n_bottleneck, n_bottleneck], init_zeros=True)
            layers.append(MaskedAffineFlow(b, t, s))
        prior = torch.distributions.MultivariateNormal(torch.zeros(n_bottleneck),
                                                       torch.eye(n_bottleneck))
        encoder_nn = MLP(hidden_units_encoder)
        encoder = NNDiagGaussian(encoder_nn)
        decoder_nn = MLP(hidden_units_decoder)
        decoder = NNDiagGaussianDecoder(decoder_nn)
        model = NormalizingFlowVAE(prior, encoder, layers, decoder)

        # Test model
        for num_samples in [1, 4]:
            with self.subTest(num_samples=num_samples):
                x = torch.randn((batch_size, n_dim))
                z, log_p, log_q = model(x, num_samples=num_samples)
                assert z.shape == (batch_size, num_samples, n_bottleneck)
                assert log_p.shape == (batch_size, num_samples)
                assert log_q.shape == (batch_size, num_samples)


if __name__ == "__main__":
    unittest.main()