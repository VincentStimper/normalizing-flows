import unittest
import torch

from normflows.flows import Radial
from normflows.flows.flow_test import FlowTest


class RadialTest(FlowTest):
    def test_radial(self):
        batch_size = 3
        for latent_size in [(2,), (5, 2), (2, 3, 4)]:
            with self.subTest(latent_size=latent_size):
                flow = Radial(latent_size)
                inputs = torch.randn((batch_size, *latent_size))
                self.checkForward(flow, inputs)


if __name__ == "__main__":
    unittest.main()