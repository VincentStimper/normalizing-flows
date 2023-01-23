import unittest
import torch

from normflows.flows.mixing import Permute, \
    LULinearPermute, InvertibleAffine
from normflows.flows.flow_test import FlowTest


class MixingTest(FlowTest):
    def test_permute(self):
        batch_size = 3
        for num_channels in [3, 4]:
            for mode in ["shuffle", "swap"]:
                with self.subTest(num_channels=num_channels, mode=mode):
                    flow = Permute(num_channels, mode=mode)
                    inputs = torch.randn((batch_size, num_channels))
                    self.checkForwardInverse(flow, inputs)

    def test_invertible_affine(self):
        batch_size = 3
        num_channels = 4
        for use_lu in [True, False]:
            with self.subTest(use_lu=use_lu):
                flow = InvertibleAffine(num_channels, use_lu=use_lu)
                inputs = torch.randn((batch_size, num_channels))
                self.checkForwardInverse(flow, inputs)

    def test_lu_linear_permute(self):
        batch_size = 3
        num_channels = 4
        for id_init in [True, False]:
            with self.subTest(id_init=id_init):
                flow = LULinearPermute(num_channels, identity_init=id_init)
                inputs = torch.randn((batch_size, num_channels))
                self.checkForwardInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()