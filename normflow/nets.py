import torch
from torch import nn

class FCN(nn.Module):
    """
    Fully Connected neural Network with ReLUs as nonlinearities
    """
    def __init__(self, hidden_units, sigmoid_output=False):
        """
        Constructor
        :param hidden_units: list of hidden units per layer
        :param sigmoid_output: add sigmoid as final output layer
        """
        super().__init__()

        self.n_hidden_layers = len(hidden_units)
        self.layers = []
        for i in range(self.n_hidden_layers - 2):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Linear(hidden_units[self.n_hidden_layers - 2], hidden_units[self.n_hidden_layers - 1]))
        if sigmoid_output:
            self.layers.append(nn.Sigmoid())

        self.layers_seq = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers_seq(x)