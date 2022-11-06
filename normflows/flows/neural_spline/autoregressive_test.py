"""
Tests for the autoregressive transforms.
Code partially taken from https://github.com/bayesiains/nsf
"""

import torch
import unittest

from normflows.flows.neural_spline import autoregressive
from normflows.flows.flow_test import FlowTest


class MaskedPiecewiseRationalQuadraticAutoregressiveFlowTest(FlowTest):
    def test_mprqas(self):
        batch_size = 5
        features = 10
        inputs = torch.rand(batch_size, features)

        flow = autoregressive.MaskedPiecewiseRationalQuadraticAutoregressive(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.checkForwardInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()
