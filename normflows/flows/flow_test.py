import unittest
import torch

from torch.testing import assert_close


class FlowTest(unittest.TestCase):
    """
    Generic test case for flow modules
    """
    def assertClose(self, actual, expected, atol=None, rtol=None):
        assert_close(actual, expected, atol=atol, rtol=rtol)

    def checkForward(self, flow, inputs, context=None):
        # Do forward transform
        if context is None:
            outputs, log_det = flow(inputs)
        else:
            outputs, log_det = flow(inputs, context)
        # Check type
        assert outputs.dtype == inputs.dtype
        # Check shape
        assert outputs.shape == inputs.shape
        # Return results
        return outputs, log_det

    def checkInverse(self, flow, inputs, context=None):
        # Do inverse transform
        if context is None:
            outputs, log_det = flow.inverse(inputs)
        else:
            outputs, log_det = flow.inverse(inputs, context)
        # Check type
        assert outputs.dtype == inputs.dtype
        # Check shape
        assert outputs.shape == inputs.shape
        # Return results
        return outputs, log_det

    def checkForwardInverse(self, flow, inputs, context=None, atol=None, rtol=None):
        # Check forward
        outputs, log_det = self.checkForward(flow, inputs, context)
        # Check inverse
        input_, log_det_ = self.checkInverse(flow, outputs, context)
        # Check identity
        self.assertClose(input_, inputs, atol, rtol)
        ld_id = log_det + log_det_
        self.assertClose(ld_id, torch.zeros_like(ld_id), atol, rtol)