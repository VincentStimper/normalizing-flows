import unittest

import torch
from torch.testing import assert_close


class FlowTest(unittest.TestCase):
    """
    Generic test case for flow modules
    """
    def assertClose(self, actual, expected):
        assert_close(actual, expected)

    def checkForward(self, flow, input):
        # Do forward transform
        output, log_det = flow(input)
        # Check type
        assert output.dtype == input.dtype
        # Check shape
        assert output.shape == input.shape
        # Return results
        return output, log_det

    def checkInverse(self, flow, input):
        # Do inverse transform
        output, log_det = flow.inverse(input)
        # Check type
        assert output.dtype == input.dtype
        # Check shape
        assert output.shape == input.shape
        # Return results
        return output, log_det

    def checkForwardInverse(self, flow, input):
        # Check forward
        output, log_det = self.checkForward(flow, input)
        # Check inverse
        input_, log_det_ = self.checkInverse(flow, output)
        # Check identity
        self.assertClose(input_, input)
        ld_id = log_det + log_det_
        self.assertClose(ld_id, torch.zeros_like(ld_id))