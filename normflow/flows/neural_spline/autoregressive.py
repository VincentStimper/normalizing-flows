"""
Implementations of autoregressive transforms.
Code taken from https://github.com/bayesiains/nsf
"""

import numpy as np
import torch
from torch.nn import functional as F

from ... import utils
from ..affine.autoregressive import Autoregressive
from normflow.nets import made as made_module
from normflow.utils import splines



class MaskedPiecewiseRationalQuadraticAutoregressive(Autoregressive):
    def __init__(self,
                 features,
                 hidden_features,
                 context_features=None,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.,
                 num_blocks=2,
                 use_residual_blocks=True,
                 random_mask=False,
                 activation=F.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.DEFAULT_MIN_DERIVATIVE
                 ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == 'linear':
            return self.num_bins * 3 - 1
        elif self.tails == 'circular':
            return self.num_bins * 3
        else:
            return self.num_bins * 3 + 1

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size,
            features,
            self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[...,:self.num_bins]
        unnormalized_heights = transform_params[...,self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[...,2*self.num_bins:]

        if hasattr(self.autoregressive_net, 'hidden_features'):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == 'linear':
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {
                'tails': self.tails,
                'tail_bound': self.tail_bound
            }
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)