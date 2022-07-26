"""
Tests for the coupling Transforms.
Code taken from https://github.com/bayesiains/nsf
"""

import itertools
import torch
import unittest

from torch import nn

from normflows import nets as nn_

from normflows.flows.neural_spline import coupling
from normflows.flows.neural_spline.flow_test import FlowTest
from normflows import utils


def create_coupling_transform(cls, shape, **kwargs):
    if len(shape) == 1:

        def create_net(in_features, out_features):
            return nn_.ResidualNet(
                in_features, out_features, hidden_features=30, num_blocks=5
            )

    else:

        def create_net(in_channels, out_channels):
            # return nn.Conv2d(in_channels, out_channels, kernel_size=1)
            return nn_.ConvResidualNet(
                in_channels=in_channels, out_channels=out_channels, hidden_channels=16
            )

    mask = utils.masks.create_mid_split_binary_mask(shape[0])

    return cls(mask=mask, transform_net_create_fn=create_net, **kwargs), mask


batch_size = 10


class PiecewiseCouplingTransformTest(FlowTest):
    classes = [coupling.PiecewiseRationalQuadraticCoupling]

    shapes = [[20], [2, 4, 4]]

    def test_forward(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape)
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_forward_unconstrained(self):
        batch_size = 10
        for shape in self.shapes:
            for cls in self.classes:
                inputs = 3.0 * torch.randn(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape, tails="linear")
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_inverse(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape)
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_inverse_unconstrained(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = 3.0 * torch.randn(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape, tails="linear")
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )

    def test_forward_inverse_are_consistent(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape)
                with self.subTest(cls=cls, shape=shape):
                    self.eps = 1e-4
                    self.assert_forward_inverse_are_consistent(transform, inputs)

    def test_forward_inverse_are_consistent_unconstrained(self):
        self.eps = 1e-5
        for shape in self.shapes:
            for cls in self.classes:
                inputs = 3.0 * torch.randn(batch_size, *shape)
                transform, mask = create_coupling_transform(cls, shape, tails="linear")
                with self.subTest(cls=cls, shape=shape):
                    self.eps = 1e-4
                    self.assert_forward_inverse_are_consistent(transform, inputs)

    def test_forward_unconditional(self):
        for shape in self.shapes:
            for cls in self.classes:
                inputs = torch.rand(batch_size, *shape)
                img_shape = shape[1:] if len(shape) > 1 else None
                transform, mask = create_coupling_transform(
                    cls, shape, apply_unconditional_transform=True, img_shape=img_shape
                )
                outputs, logabsdet = transform(inputs)
                with self.subTest(cls=cls, shape=shape):
                    self.assert_tensor_is_good(outputs, [batch_size] + shape)
                    self.assert_tensor_is_good(logabsdet, [batch_size])
                    self.assertNotEqual(
                        outputs[:, mask <= 0, ...], inputs[:, mask <= 0, ...]
                    )


if __name__ == "__main__":
    unittest.main()
