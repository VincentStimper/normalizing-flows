import torch
import unittest

from normflows.flows.affine import autoregressive
from normflows.flows.neural_spline.flow_test import FlowTest


class MaskedAffineAutoregressiveTest(FlowTest):
    def test_forward(self):
        batch_size = 10
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
                transform = autoregressive.MaskedAffineAutoregressive(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_inverse(self):
        batch_size = 10
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
                transform = autoregressive.MaskedAffineAutoregressive(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                outputs, logabsdet = transform.inverse(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        features = 20
        inputs = torch.randn(batch_size, features)
        self.eps = 1e-6
        for use_residual_blocks, random_mask in [
            (False, False),
            (False, True),
            (True, False),
        ]:
            with self.subTest(
                use_residual_blocks=use_residual_blocks, random_mask=random_mask
            ):
                transform = autoregressive.MaskedAffineAutoregressive(
                    features=features,
                    hidden_features=30,
                    num_blocks=5,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=random_mask,
                )
                self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()
