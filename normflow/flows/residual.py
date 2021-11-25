from .base import Flow

# Try importing Residual Flow dependencies
try:
    from residual_flows.layers import iResBlock
except:
    print('Warning: Dependencies for Residual Flows could '
          'not be loaded. Other models can still be used.')



class Residual(Flow):
    """
    Invertible residual net block, wrapper to the implementation of Chen et al.,
    see https://github.com/rtqichen/residual-flows
    """
    def __init__(self, net, n_exact_terms=2, n_samples=1, reduce_memory=True,
                 reverse=True):
        """
        Constructor
        :param net: Neural network, must be Lipschitz continuous with L < 1
        :param n_exact_terms: Number of terms always included in the power series
        :param n_samples: Number of samples used to estimate power series
        :param reduce_memory: Flag, if true Neumann series and precomputations
        for backward pass in forward pass are done
        :param reverse: Flag, if true the map f(x) = x + net(x) is applied in
        the inverse pass, otherwise it is done in forward
        """
        super().__init__()
        self.reverse = reverse
        self.iresblock = iResBlock(net, n_samples=n_samples,
                                   n_exact_terms=n_exact_terms,
                                   neumann_grad=reduce_memory,
                                   grad_in_forward=reduce_memory)

    def forward(self, z):
        if self.reverse:
            z, log_det = self.iresblock.inverse(z, 0)
        else:
            z, log_det = self.iresblock.forward(z, 0)
        return z, -log_det.view(-1)

    def inverse(self, z):
        if self.reverse:
            z, log_det = self.iresblock.forward(z, 0)
        else:
            z, log_det = self.iresblock.inverse(z, 0)
        return z, -log_det.view(-1)