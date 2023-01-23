import unittest
import torch

from normflows.flows.base import Reverse, Composite
from normflows.flows.affine.coupling import AffineConstFlow
from normflows.flows.flow_test import FlowTest


class BaseTest(FlowTest):
    def test_reverse(self):
        batch_size = 5
        for n_dim in [2, 7]:
            with self.subTest(n_dim=n_dim):
                flow = AffineConstFlow((n_dim,))
                flow_rev = Reverse(flow)
                inputs = torch.randn((batch_size, n_dim))
                self.checkForwardInverse(flow_rev, inputs)
                outputs = flow(inputs)
                outputs_ = flow_rev(inputs)
                self.assertClose(outputs, outputs_)
                outputs = flow.inverse(inputs)
                outputs_ = flow_rev.inverse(inputs)
                self.assertClose(outputs, outputs_)

    def test_composite(self):
        batch_size = 5
        for n_dim in [2, 7]:
            with self.subTest(n_dim=n_dim):
                flows = [AffineConstFlow((n_dim,)),
                         AffineConstFlow((n_dim,))]
                flow = Composite(flows)
                inputs = torch.randn((batch_size, n_dim))
                self.checkForwardInverse(flow, inputs)



if __name__ == "__main__":
    unittest.main()