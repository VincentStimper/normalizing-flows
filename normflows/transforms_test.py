import unittest
import torch

from normflows.transforms import Logit, Shift
from normflows.flows.flow_test import FlowTest


class TransformsTest(FlowTest):
    def test_transforms(self):
        batch_size = 5
        transforms = [Logit, Shift]
        for shape in [(1,), (2, 3)]:
            for transform_c in transforms:
                with self.subTest(shape=shape,
                                  transform_c=transform_c):
                    transform = transform_c()
                    inputs = torch.randn((batch_size,) + shape)
                    self.checkForwardInverse(transform, inputs)


if __name__ == "__main__":
    unittest.main()