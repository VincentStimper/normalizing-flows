import unittest
import torch
import numpy as np

from normflows.flows.periodic import PeriodicWrap, PeriodicShift
from normflows.flows.flow_test import FlowTest


class PeriodicTest(FlowTest):
    def test_periodic_wrap(self):
        batch_size = 3
        n_dim = 4
        for bound in [1.0, np.pi]:
            for ind in [0, [1], [2, 3]]:
                with self.subTest(bound=bound, ind=ind):
                    flow = PeriodicWrap(ind, bound=bound)
                    inputs = torch.rand((batch_size, n_dim)) * bound
                    self.checkForwardInverse(flow, inputs)
                    inputs[:, ind] = inputs[:, ind] + bound
                    outputs, _ = flow.inverse(inputs)
                    assert torch.all(outputs[:, ind] < bound)

    def test_periodic_shift(self):
        batch_size = 3
        n_dim = 4
        for bound in [1.0, np.pi]:
            for ind in [0, [1], [2, 3]]:
                with self.subTest(bound=bound, ind=ind):
                    flow = PeriodicShift(ind, bound=bound, shift=bound / 3)
                    inputs = torch.rand((batch_size, n_dim)) * bound
                    self.checkForwardInverse(flow, inputs)
                    inputs = inputs * 2 - bound
                    self.checkForwardInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()