from .base import Flow


class Reverse(Flow):
    """
    Switches the forward transform of a flow layer with its inverse and vice versa
    """
    def __init__(self, flow):
        """
        Constructor
        :param flow: Flow layer to be reversed
        """
        super().__init__()
        self.flow = flow

    def forward(self, z):
        return self.flow.inverse(z)

    def inverse(self, z):
        return self.flow.forward(z)