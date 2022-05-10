import torch
from torch import nn
import numpy as np

from ..base import Flow
from .coupling import PiecewiseRationalQuadraticCoupling
from .autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from ...nets.resnet import ResidualNet
from ...utils.masks import create_alternating_binary_mask
from ...utils.nn import PeriodicFeatures
from ...utils.splines import DEFAULT_MIN_DERIVATIVE



class CoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see https://github.com/bayesiains/nsf
    """
    def __init__(
            self,
            num_input_channels,
            num_blocks,
            num_hidden_channels,
            num_bins=8,
            tails='linear',
            tail_bound=3.,
            activation=nn.ReLU,
            dropout_probability=0.,
            reverse_mask=False
    ):
        """
        Constructor
        :param num_input_channels: Flow dimension
        :type num_input_channels: Int
        :param num_blocks: Number of residual blocks of the parameter NN
        :type num_blocks: Int
        :param num_hidden_channels: Number of hidden units of the NN
        :type num_hidden_channels: Int
        :param num_bins: Number of bins
        :type num_bins: Int
        :param tails: Behaviour of the tails of the distribution,
        can be linear, circular for periodic distribution, or None for
        distribution on the compact interval
        :type tails: String
        :param tail_bound: Bound of the spline tails
        :type tail_bound: Float
        :param activation: Activation function
        :type activation: torch module
        :param dropout_probability: Dropout probability of the NN
        :type dropout_probability: Float
        :param reverse_mask: Flag whether the reverse mask should be used
        :type reverse_mask: Boolean
        """
        super().__init__()

        def transform_net_create_fn(in_features, out_features):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=None,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False
            )

        self.prqct=PiecewiseRationalQuadraticCoupling(
            mask=create_alternating_binary_mask(
                num_input_channels,
                even=reverse_mask
            ),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,

            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
            apply_unconditional_transform=True
        )

    def forward(self, z):
        z, log_det = self.prqct.inverse(z)
        return z, log_det.view(-1)

    def inverse(self, z):
        z, log_det = self.prqct(z)
        return z, log_det.view(-1)


class CircularCoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer with circular coordinates
    """
    def __init__(
            self,
            num_input_channels,
            num_blocks,
            num_hidden_channels,
            ind_circ,
            num_bins=8,
            tail_bound=3.,
            activation=nn.ReLU,
            dropout_probability=0.,
            reverse_mask=False,
            mask=None,
            init_identity=True
    ):
        """
        Constructor
        :param num_input_channels: Flow dimension
        :type num_input_channels: Int
        :param num_blocks: Number of residual blocks of the parameter NN
        :type num_blocks: Int
        :param num_hidden_channels: Number of hidden units of the NN
        :type num_hidden_channels: Int
        :param ind_circ: Indices of the circular coordinates
        :type ind_circ: Iterable
        :param num_bins: Number of bins
        :type num_bins: Int
        :param tail_bound: Bound of the spline tails
        :type tail_bound: Float or Iterable
        :param activation: Activation function
        :type activation: torch module
        :param dropout_probability: Dropout probability of the NN
        :type dropout_probability: Float
        :param reverse_mask: Flag whether the reverse mask should be used
        :type reverse_mask: Boolean
        :param mask: Mask to be used, alternating masked generated is None
        :param mask: torch tensor
        :param init_identity: Flag, initialize transform as identity
        :type init_identity: Boolean
        """
        super().__init__()

        if mask is None:
            mask = create_alternating_binary_mask(num_input_channels,
                                                  even=reverse_mask)
        features_vector = torch.arange(num_input_channels)
        identity_features = features_vector.masked_select(mask <= 0)
        ind_circ = torch.tensor(ind_circ)
        ind_circ_id = []
        for i, id in enumerate(identity_features):
            if id in ind_circ:
                ind_circ_id += [i]

        if torch.is_tensor(tail_bound):
            scale_pf = np.pi / tail_bound[ind_circ_id]
        else:
            scale_pf = np.pi / tail_bound

        def transform_net_create_fn(in_features, out_features):
            if len(ind_circ_id) > 0:
                pf = PeriodicFeatures(in_features, ind_circ_id, scale_pf)
            else:
                pf = None
            net = ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=None,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
                preprocessing=pf
            )
            if init_identity:
                torch.nn.init.constant_(net.final_layer.weight, 0.)
                torch.nn.init.constant_(net.final_layer.bias,
                                        np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1))
            return net


        tails = ['circular' if i in ind_circ else 'linear'
                 for i in range(num_input_channels)]

        self.prqct=PiecewiseRationalQuadraticCoupling(
            mask=mask,
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            apply_unconditional_transform=True
        )

    def forward(self, z):
        z, log_det = self.prqct.inverse(z)
        return z, log_det.view(-1)

    def inverse(self, z):
        z, log_det = self.prqct(z)
        return z, log_det.view(-1)


class AutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see https://github.com/bayesiains/nsf
    """
    def __init__(
            self,
            num_input_channels,
            num_blocks,
            num_hidden_channels,
            num_bins=8,
            tail_bound=3,
            activation=nn.ReLU,
            dropout_probability=0.,
            permute_mask=False,
            init_identity=True
    ):
        """
        Constructor
        :param num_input_channels: Flow dimension
        :type num_input_channels: Int
        :param num_blocks: Number of residual blocks of the parameter NN
        :type num_blocks: Int
        :param num_hidden_channels: Number of hidden units of the NN
        :type num_hidden_channels: Int
        :param num_bins: Number of bins
        :type num_bins: Int
        :param tail_bound: Bound of the spline tails
        :type tail_bound: Int
        :param activation: Activation function
        :type activation: torch module
        :param dropout_probability: Dropout probability of the NN
        :type dropout_probability: Float
        :param permute_mask: Flag, permutes the mask of the NN
        :type permute_mask: Boolean
        :param init_identity: Flag, initialize transform as identity
        :type init_identity: Boolean
        """
        super().__init__()

        self.mprqat=MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=None,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity)

    def forward(self, z):
        z, log_det = self.mprqat.inverse(z)
        return z, log_det.view(-1)

    def inverse(self, z):
        z, log_det = self.mprqat(z)
        return z, log_det.view(-1)


class CircularAutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see https://github.com/bayesiains/nsf
    """
    def __init__(
            self,
            num_input_channels,
            num_blocks,
            num_hidden_channels,
            ind_circ,
            num_bins=8,
            tail_bound=3,
            activation=nn.ReLU,
            dropout_probability=0.,
            permute_mask=True,
            init_identity=True
    ):
        """
        Constructor
        :param num_input_channels: Flow dimension
        :type num_input_channels: Int
        :param num_blocks: Number of residual blocks of the parameter NN
        :type num_blocks: Int
        :param num_hidden_channels: Number of hidden units of the NN
        :type num_hidden_channels: Int
        :param ind_circ: Indices of the circular coordinates
        :type ind_circ: Iterable
        :param num_bins: Number of bins
        :type num_bins: Int
        :param tail_bound: Bound of the spline tails
        :type tail_bound: Int
        :param activation: Activation function
        :type activation: torch module
        :param dropout_probability: Dropout probability of the NN
        :type dropout_probability: Float
        :param permute_mask: Flag, permutes the mask of the NN
        :type permute_mask: Boolean
        :param init_identity: Flag, initialize transform as identity
        :type init_identity: Boolean
        """
        super().__init__()

        tails = ['circular' if i in ind_circ else 'linear'
                 for i in range(num_input_channels)]

        self.mprqat=MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=None,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity)

    def forward(self, z):
        z, log_det = self.mprqat.inverse(z)
        return z, log_det.view(-1)

    def inverse(self, z):
        z, log_det = self.mprqat(z)
        return z, log_det.view(-1)