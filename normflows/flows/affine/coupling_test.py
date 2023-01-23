import unittest
import torch

from normflows.flows import MaskedAffineFlow, CCAffineConst
from normflows.nets import MLP
from normflows.flows.flow_test import FlowTest


class CouplingTest(FlowTest):
    def test_mask_affine(self):
        batch_size = 5
        for latent_size in [2, 7]:
            with self.subTest(latent_size=latent_size):
                b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
                s = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                t = MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                flow = MaskedAffineFlow(b, t, s)
                inputs = torch.randn((batch_size, latent_size))
                self.checkForwardInverse(flow, inputs)

    def test_cc_affine(self):
        batch_size = 5
        for shape in [(5,), (2, 3, 4)]:
            for num_classes in [2, 5]:
                with self.subTest(shape=shape, num_classes=num_classes):
                    flow = CCAffineConst(shape, num_classes)
                    x = torch.randn((batch_size,) + shape)
                    y = torch.rand((batch_size,) + (num_classes,))
                    x_, log_det = flow(x, y)
                    x__, log_det_ = flow(x_, y)

                    assert x_.dtype == x.dtype
                    assert x__.dtype == x.dtype

                    assert x_.shape == x.shape
                    assert x__.shape == x.shape

                    self.assertClose(x__, x)
                    id_ld = log_det + log_det_
                    self.assertClose(id_ld, torch.zeros_like(id_ld))



if __name__ == "__main__":
    unittest.main()