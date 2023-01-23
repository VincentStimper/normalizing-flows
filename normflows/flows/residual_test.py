import unittest
import torch

from normflows.flows import Residual
from normflows.nets import LipschitzMLP, LipschitzCNN
from normflows.utils.optim import update_lipschitz
from normflows.flows.flow_test import FlowTest


class ResidualTest(FlowTest):
    def test_residual_mlp(self):
        batch_size = 3
        hidden_units = 128
        hidden_layers = 2
        params = [(2, False, True, 'geometric', True),
                  (2, True, True, 'poisson', False),
                  (4, True, False, 'geometric', False),
                  (5, False, False, 'poisson', False)]
        for latent_size, reduce_memory, exact_trace, n_dist, brute_force in params:
            with self.subTest(latent_size=latent_size, reduce_memory=reduce_memory,
                              exact_trace=exact_trace, n_dist=n_dist,
                              brute_force=brute_force):
                layer = [latent_size] + [hidden_units] * hidden_layers + [latent_size]
                net = LipschitzMLP(layer, init_zeros=exact_trace,
                                   lipschitz_const=0.9)
                flow = Residual(net, reduce_memory=reduce_memory, n_dist=n_dist,
                                exact_trace=exact_trace, brute_force=brute_force)
                inputs = torch.randn((batch_size, latent_size))
                if exact_trace:
                    self.checkForwardInverse(flow, inputs, atol=1e-4, rtol=1e-4)
                else:
                    outputs, _ = self.checkForward(flow, inputs)
                    inputs_, _ = self.checkInverse(flow, outputs)
                    self.assertClose(inputs, inputs_, atol=1e-4, rtol=1e-4)
                update_lipschitz(flow, 1)
                _ = net.net[1].compute_one_iter()



    def test_residual_cnn(self):
        batch_size = 1
        hidden_units = 128
        params = [(1, 3, False), (3, 4, True)]
        img_size = (4, 4)
        for kernel_size, latent_size, reduce_memory in params:
            with self.subTest(latent_size=latent_size, reduce_memory=reduce_memory):
                channels = [latent_size, hidden_units, latent_size]
                net = LipschitzCNN(channels, 2 * [kernel_size])
                flow = Residual(net, reduce_memory=reduce_memory)
                inputs = torch.randn((batch_size, latent_size, *img_size))
                outputs, log_det = self.checkForward(flow, inputs)
                inputs_, _ = self.checkInverse(flow, outputs)
                self.assertClose(inputs, inputs_, atol=1e-4, rtol=1e-4)
                loss = torch.mean(log_det)
                loss.backward()
                update_lipschitz(flow, 1)
                _ = net.net[1].compute_one_iter()


if __name__ == "__main__":
    unittest.main()