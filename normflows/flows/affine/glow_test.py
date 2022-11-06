import unittest
import torch

from normflows.flows import GlowBlock
from normflows.flows.flow_test import FlowTest


class GlowTest(FlowTest):
    def test_glow(self):
        img_size = (4, 4)
        hidden_channels = 8
        for batch_size, channels, scale, split_mode, use_lu, net_actnorm in [
            (1, 3, True, "channel", True, False),
            (2, 3, True, "channel_inv", True, False),
            (1, 4, True, "channel_inv", True, True),
            (2, 4, True, "channel", True, False),
            (1, 4, False, "channel", False, False),
            (1, 4, True, "checkerboard", True, True),
            (3, 5, False, "checkerboard", False, True)
        ]:
            with self.subTest(batch_size=batch_size, channels=channels,
                              scale=scale, split_mode=split_mode,
                              use_lu=use_lu, net_actnorm=net_actnorm):
                inputs = torch.rand((batch_size, channels) + img_size)
                flow = GlowBlock(channels, hidden_channels,
                                 scale=scale, split_mode=split_mode,
                                 use_lu=use_lu, net_actnorm=net_actnorm)
                self.checkForwardInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()