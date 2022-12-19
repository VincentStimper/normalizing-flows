import unittest
import torch

from torch.testing import assert_close


class FlowTest(unittest.TestCase):
    """
    Generic test case for flow modules
    """
    def assertClose(self, actual, expected, atol=None, rtol=None):
        assert_close(actual, expected, atol=atol, rtol=rtol)

    def checkForward(self, flow, inputs):
        # Do forward transform
        outputs, log_det = flow(inputs)
        # Check type
        assert outputs.dtype == inputs.dtype
        # Check shape
        assert outputs.shape == inputs.shape
        # Return results
        return outputs, log_det

    def checkInverse(self, flow, inputs):
        # Do inverse transform
        outputs, log_det = flow.inverse(inputs)
        # Check type
        assert outputs.dtype == inputs.dtype
        # Check shape
        assert outputs.shape == inputs.shape
        # Return results
        return outputs, log_det

    def checkForwardInverse(self, flow, inputs, atol=None, rtol=None):
        # Check forward
        outputs, log_det = self.checkForward(flow, inputs)
        # Check inverse
        input_, log_det_ = self.checkInverse(flow, outputs)
        # Check identity
        self.assertClose(input_, inputs, atol, rtol)
        ld_id = log_det + log_det_
        self.assertClose(ld_id, torch.zeros_like(ld_id), atol, rtol)