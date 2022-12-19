"""
Tests for MADE.
Code partially taken from https://github.com/bayesiains/nsf
"""

import torch
import unittest

from normflows.nets import made
from torch.testing import assert_close


class ShapeTest(unittest.TestCase):
    def test_unconditional(self):
        features = 10
        hidden_features = 20
        num_blocks = 3
        output_multiplier = 3
        batch_size = 4

        inputs = torch.randn(batch_size, features)

        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                model = made.MADE(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    output_multiplier=output_multiplier,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs = model(inputs)
                assert outputs.dim() == 2
                assert outputs.shape[0] == batch_size
                assert outputs.shape[1] == output_multiplier * features


class ConnectivityTest(unittest.TestCase):
    def test_gradients(self):
        features = 10
        hidden_features = 32
        num_blocks = 5
        output_multiplier = 3

        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                model = made.MADE(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    output_multiplier=output_multiplier,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                inputs = torch.randn(1, features)
                inputs.requires_grad = True
                for k in range(features * output_multiplier):
                    outputs = model(inputs)
                    outputs[0, k].backward()
                    depends = inputs.grad.data[0] != 0.0
                    dim = k // output_multiplier
                    assert torch.all(depends[dim:] == 0) == 1

    def test_total_mask_sequential(self):
        features = 10
        hidden_features = 32
        num_blocks = 5
        output_multiplier = 1

        for use_residual_blocks in [True, False]:
            with self.subTest(use_residual_blocks=use_residual_blocks):
                model = made.MADE(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    output_multiplier=output_multiplier,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=False,
                )
                total_mask = model.initial_layer.mask
                for block in model.blocks:
                    if use_residual_blocks:
                        self.assertIsInstance(block, made.MaskedResidualBlock)
                        total_mask = block.linear_layers[0].mask @ total_mask
                        total_mask = block.linear_layers[1].mask @ total_mask
                    else:
                        self.assertIsInstance(block, made.MaskedFeedforwardBlock)
                        total_mask = block.linear.mask @ total_mask
                total_mask = model.final_layer.mask @ total_mask
                total_mask = (total_mask > 0).float()
                reference = torch.tril(torch.ones([features, features]), -1)
                assert_close(total_mask, reference)

    def test_total_mask_random(self):
        features = 10
        hidden_features = 32
        num_blocks = 5
        output_multiplier = 1

        model = made.MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            output_multiplier=output_multiplier,
            use_residual_blocks=False,
            random_mask=True,
        )
        total_mask = model.initial_layer.mask
        for block in model.blocks:
            assert isinstance(block, made.MaskedFeedforwardBlock)
            total_mask = block.linear.mask @ total_mask
        total_mask = model.final_layer.mask @ total_mask
        total_mask = (total_mask > 0).float()
        assert_close(torch.triu(total_mask), torch.zeros([features, features]))


if __name__ == "__main__":
    unittest.main()
