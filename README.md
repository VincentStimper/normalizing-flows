# Normalizing Flows

[![Code Style: Black](https://img.shields.io/badge/Code%20Style-black-black.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/PyPI-1.4-blue.svg)](https://pypi.org/project/normflows/)


This is a PyTorch implementation of several normalizing flows, including 
a variational autoencoder. It is used in the articles 
[A Gradient Based Strategy for Hamiltonian Monte Carlo Hyperparameter Optimization](https://proceedings.mlr.press/v139/campbell21a.html)
and [Resampling Base Distributions of Normalizing Flows](https://arxiv.org/abs/2110.15828).


## Implemented Flows

* Planar Flow ([Rezende & Mohamed, 2015](https://arxiv.org/abs/1505.05770))
* Radial Flow ([Rezende & Mohamed, 2015](https://arxiv.org/abs/1505.05770))
* NICE ([Dinh et al., 2014](https://arxiv.org/abs/1410.8516))
* Real NVP ([Dinh et al., 2016](https://arxiv.org/abs/1605.08803))
* Glow ([Kingma & Dhariwal, 2018](https://arxiv.org/abs/1807.03039))
* Masked Autoregressive Flow ([Papamakarios et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html))
* Neural Spline Flow ([Durkan et al., 2019](https://arxiv.org/abs/1906.04032))
* Circular Neural Spline Flow ([Rezende et al., 2020](http://proceedings.mlr.press/v119/rezende20a.html))
* Residual Flow ([Chen et al., 2019](https://arxiv.org/abs/1906.02735))
* Stochastic Normalizing Flows ([Wu et al., 2020](https://arxiv.org/abs/2002.06707))

Note that Neural Spline Flows with circular and non-circular coordinates
are also supported.

## Methods of Installation

The latest version of the package can be installed via pip

```
pip install normflows
```

At least Python 3.7 is required. If you want to use a GPU, make sure that
PyTorch is set up correctly by following the instructions at the
[PyTorch website](https://pytorch.org/get-started/locally/).

To run the example notebooks clone the repository first

```
git clone https://github.com/VincentStimper/normalizing-flows.git
```

and then install the dependencies.

```
pip install -r requirements_examples.txt
```

## Usage

A normalizing flow consists of a base distribution, defined in 
[`nf.distributions.base`](https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/distributions/base.py),
and a list of flows, given in
[`nf.flows`](https://github.com/VincentStimper/normalizing-flows/tree/master/normflows/flows).
Let's assume our target is a 2D distribution. We pick a diagonal Gaussian
base distribution, which is the most popular choice. Our flow shall be a
[Real NVP model](https://arxiv.org/abs/1605.08803) and, therefore, we need
to define a neural network for computing the parameters of the affine coupling
map. One dimension is used to compute the scale and shift parameter for the
other dimension. After each coupling layer we swap their roles.

```python
import normflows as nf

# Define 2D base distribution
base = nf.distributions.base.DiagGaussian(2)

# Define list of flows
num_layers = 16
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 32 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 32, 32, 2], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))
```

Once they are set up, we can define a
[`nf.NormalizingFlow`](https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py#L7)
model. If the target density is available, it can be added to the model
to be used during training. Sample target distributions are given in
[`nf.distributions.target`](https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/distributions/target.py).

```python
# If the target density is not given
model = nf.NormalizingFlow(base, flows)

# If the target density is given
target = nf.distributions.target.TwoMoons()
model = nf.NormalizingFlow(base, flows, target)
```

The loss can be computed with the methods of the model and minimized.

```python
# When doing maximum likelihood learning, i.e. minimizing the forward KLD
# with no target distribution given
loss = model.forward_kld(x)

# When minimizing the reverse KLD based on the given target distribution
loss = model.reverse_kld(num_samples=1024)

# Optimization as usual
loss.backward()
optimizer.step()
```

For more illustrative examples of how to use the package see the
[`example`](https://github.com/VincentStimper/normalizing-flows/tree/master/example)
directory. More advanced experiments can be done with the scripts listed in the
[repository about resampled base distributions](https://github.com/VincentStimper/resampled-base-flows),
see its [`experiments`](https://github.com/VincentStimper/resampled-base-flows/tree/master/experiments)
folder.





