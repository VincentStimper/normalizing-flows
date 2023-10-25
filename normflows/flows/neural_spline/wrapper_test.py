import unittest
import torch
import numpy as np

from normflows.flows import CoupledRationalQuadraticSpline, \
    AutoregressiveRationalQuadraticSpline, \
    CircularCoupledRationalQuadraticSpline, \
    CircularAutoregressiveRationalQuadraticSpline
from normflows.flows.flow_test import FlowTest


class NsfWrapperTest(FlowTest):
    def test_normal_nsf(self):
        batch_size = 3
        hidden_units = 128
        hidden_layers = 2
        for latent_size in [2, 5]:
            for flow_cls in [CoupledRationalQuadraticSpline,
                             AutoregressiveRationalQuadraticSpline]:
                for context_feature in [None, 3]:
                    with self.subTest(latent_size=latent_size, flow_cls=flow_cls):
                        flow = flow_cls(latent_size, hidden_units, hidden_layers,
                                        num_context_channels=context_feature)
                        inputs = torch.randn((batch_size, latent_size))
                        if context_feature is None:
                            context = None
                        else:
                            context = torch.randn((batch_size, context_feature))
                        self.checkForwardInverse(flow, inputs, context)

    def test_circular_nsf(self):
        batch_size = 3
        hidden_units = 128
        hidden_layers = 2
        params = [(2, [1], torch.tensor([5., np.pi])),
                  (5, [0, 3], torch.tensor([np.pi, 5., 4., 6., 3.])),
                  (2, [1], torch.tensor([5., np.pi]))]
        for latent_size, ind_circ, tail_bound in params:
            for flow_cls in [CircularCoupledRationalQuadraticSpline,
                             CircularAutoregressiveRationalQuadraticSpline]:
                for context_feature in [None, 3]:
                    with self.subTest(latent_size=latent_size, ind_circ=ind_circ,
                                      tail_bound=tail_bound, flow_cls=flow_cls,
                                      context_feature=context_feature):
                        flow = flow_cls(latent_size, hidden_units, hidden_layers,
                                        ind_circ, tail_bound=tail_bound,
                                        num_context_channels=context_feature)
                        inputs = 6 * torch.rand((batch_size, latent_size)) - 3
                        if context_feature is None:
                            context = None
                        else:
                            context = torch.randn((batch_size, context_feature))
                        self.checkForwardInverse(flow, inputs, context)


if __name__ == "__main__":
    unittest.main()