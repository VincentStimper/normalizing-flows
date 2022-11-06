import torch
import unittest

from normflows.flows.affine import autoregressive
from normflows.flows.flow_test import FlowTest


class MaskedAffineAutoregressiveTest(FlowTest):
    def test_maf(self):
        batch_size = 3
        features = 20
        inputs = torch.randn(batch_size, features)
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                flow = autoregressive.MaskedAffineAutoregressive(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                self.checkForwardInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()
