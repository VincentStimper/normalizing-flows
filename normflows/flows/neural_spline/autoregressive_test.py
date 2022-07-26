"""
Tests for the autoregressive transforms.
Code taken from https://github.com/bayesiains/nsf
"""

import torch
import unittest

from normflows.flows.neural_spline import autoregressive
from normflows.flows.neural_spline.flow_test import FlowTest


class MaskedPiecewiseRationalQuadraticAutoregressiveFlowTest(FlowTest):
    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.rand(batch_size, features)
        self.eps = 1e-3

        transform = autoregressive.MaskedPiecewiseRationalQuadraticAutoregressive(
            num_bins=10,
            features=features,
            hidden_features=30,
            num_blocks=5,
            use_residual_blocks=True,
        )

        self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()
