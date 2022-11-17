import unittest
import torch

from normflows.flows import Residual
from normflows.nets import LipschitzMLP, LipschitzCNN
from normflows.flows.flow_test import FlowTest


class ResidualTest(FlowTest):
    def test_residual_mlp(self):
        batch_size = 3
        hidden_units = 128
        hidden_layers = 2
        for latent_size in [2, 5]:
            for reduce_memory in [True, False]:
                for exact_trace in [True, False]:
                    with self.subTest(latent_size=latent_size, reduce_memory=reduce_memory,
                                      exact_trace=exact_trace):
                        layer = [latent_size] + [hidden_units] * hidden_layers + [latent_size]
                        net = LipschitzMLP(layer, init_zeros=exact_trace, lipschitz_const=0.9)
                        flow = Residual(net, reduce_memory=reduce_memory,
                                        exact_trace=exact_trace)
                        inputs = torch.randn((batch_size, latent_size))
                        if exact_trace:
                            self.checkForwardInverse(flow, inputs, atol=1e-4, rtol=1e-4)
                        else:
                            outputs, _ = self.checkForward(flow, inputs)
                            inputs_, _ = self.checkInverse(flow, outputs)
                            self.assertClose(inputs, inputs_, atol=1e-4, rtol=1e-4)



    def test_residual_cnn(self):
        batch_size = 1
        hidden_units = 128
        kernel_size = 1
        img_size = (4, 4)
        for latent_size in [3, 4]:
            for reduce_memory in [True, False]:
                with self.subTest(latent_size=latent_size, reduce_memory=reduce_memory):
                    channels = [latent_size, hidden_units, latent_size]
                    net = LipschitzCNN(channels, 2 * [kernel_size])
                    flow = Residual(net, reduce_memory=reduce_memory)
                    inputs = torch.randn((batch_size, latent_size, *img_size))
                    outputs, _ = self.checkForward(flow, inputs)
                    inputs_, _ = self.checkInverse(flow, outputs)
                    self.assertClose(inputs, inputs_, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()