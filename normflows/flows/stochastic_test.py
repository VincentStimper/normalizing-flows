import unittest
import torch
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal

from normflows.flows.stochastic import MetropolisHastings, \
    HamiltonianMonteCarlo
from normflows.distributions.mh_proposal import DiagGaussianProposal
from normflows.flows.flow_test import FlowTest


class StochasticTest(FlowTest):
    def test_metropolis_hastings(self):
        batch_size = 3
        # Set up flow
        n_dim = 4
        target = MultivariateNormal(torch.zeros(n_dim), torch.eye(n_dim))
        proposal = DiagGaussianProposal((n_dim,), 0.1)
        flow = MetropolisHastings(target, proposal, 3)
        # Run tests
        inputs = torch.randn((batch_size, n_dim))
        _ = self.checkForward(flow, inputs)
        _ = self.checkInverse(flow, inputs)

    def test_hamiltonian_monte_carlo(self):
        batch_size = 3
        # Set up flow
        n_dim = 4
        target = MultivariateNormal(torch.zeros(n_dim), torch.eye(n_dim))
        flow = HamiltonianMonteCarlo(target, 3, -2. * torch.ones(n_dim),
                                     torch.zeros(n_dim))
        # Run tests
        inputs = torch.randn((batch_size, n_dim))
        _ = self.checkForward(flow, inputs)
        _ = self.checkInverse(flow, inputs)


if __name__ == "__main__":
    unittest.main()