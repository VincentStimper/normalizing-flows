# `normflows`: A PyTorch Package for Normalizing Flows

[![documentation](https://github.com/VincentStimper/normalizing-flows/actions/workflows/mkdocs.yaml/badge.svg)](https://vincentstimper.github.io/normalizing-flows/)
![unit-tests](https://github.com/VincentStimper/normalizing-flows/actions/workflows/pytest.yaml/badge.svg)
![code coverage](https://raw.githubusercontent.com/VincentStimper/normalizing-flows/coverage-badge/coverage.svg?raw=true)
[![License: MIT](https://img.shields.io/badge/Licence-MIT-b31b1b.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05361/status.svg)](https://doi.org/10.21105/joss.05361)
[![PyPI](https://img.shields.io/badge/PyPI-1.7.2-blue.svg)](https://pypi.org/project/normflows/)
[![Downloads](https://static.pepy.tech/personalized-badge/normflows?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/normflows)


`normflows` is a PyTorch implementation of discrete normalizing flows. Many popular flow architectures are implemented,
see the [list below](#implemented-flows). The package can be easily [installed via pip](#installation).
The basic usage is described [here](#usage), and a [full documentation](https://vincentstimper.github.io/normalizing-flows/) 
is available as well. A more detailed description of this package is given in our
[accompanying paper](https://joss.theoj.org/papers/10.21105/joss.05361).

Several sample use cases are provided in the 
[`examples` folder](https://github.com/VincentStimper/normalizing-flows/blob/master/examples), 
including [Glow](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/glow.ipynb),
a [VAE](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/vae.py), and
a [Residual Flow](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/residual.ipynb).
Moreover, two simple applications are highlighed in the [examples section](#examples). You can run them 
yourself in Google Colab using the links below to get a feeling for `normflows`.

| Link                                                                                                                                                                                                                                                  | Description                                                             |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| <a href="https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/real_nvp_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>          | Real NVP applied to a 2D bimodal target distribution                    |
| <a href="https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/paper_example_nsf_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Modeling a distribution on a cylinder surface with a neural spline flow |
| <a href="https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/glow_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>              | Modeling and generating CIFAR-10 images with Glow                       |


## Implemented Flows

| Architecture | Reference                                                                                                                 |
|--------------|---------------------------------------------------------------------------------------------------------------------------|
| Planar Flow  | [Rezende & Mohamed, 2015](http://proceedings.mlr.press/v37/rezende15.html)                                                |
| Radial Flow  | [Rezende & Mohamed, 2015](http://proceedings.mlr.press/v37/rezende15.html)                                                |
| NICE         | [Dinh et al., 2014](https://arxiv.org/abs/1410.8516)                                                                      |
| Real NVP     | [Dinh et al., 2017](https://openreview.net/forum?id=HkpbnH9lx)                                                            |
| Glow         | [Kingma et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html)                                                                   |
| Masked Autoregressive Flow | [Papamakarios et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html) |
| Neural Spline Flow | [Durkan et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html)                                                                    |
| Circular Neural Spline Flow | [Rezende et al., 2020](http://proceedings.mlr.press/v119/rezende20a.html)                                                 |
| Residual Flow | [Chen et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/5d0d5594d24f0f955548f0fc0ff83d10-Abstract.html)                                                                     |
| Stochastic Normalizing Flow | [Wu et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/41d80bfc327ef980528426fc810a6d7a-Abstract.html)                                                                       |

Note that Neural Spline Flows with circular and non-circular coordinates
are supported as well.

## Installation

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
[`nf.flows`](https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/flows).
Let's assume our target is a 2D distribution. We pick a diagonal Gaussian
base distribution, which is the most popular choice. Our flow shall be a
[Real NVP model](https://openreview.net/forum?id=HkpbnH9lx) and, therefore, we need
to define a neural network for computing the parameters of the affine coupling
map. One dimension is used to compute the scale and shift parameter for the
other dimension. After each coupling layer we swap their roles.

```python
import normflows as nf

# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(2)

# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))
```

Once they are set up, we can define a
[`nf.NormalizingFlow`](https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py#L9)
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
loss = model.reverse_kld(num_samples=512)

# Optimization as usual
loss.backward()
optimizer.step()
```

## Examples

We provide several illustrative examples of how to use the package in the
[`examples`](https://github.com/VincentStimper/normalizing-flows/blob/master/examples)
directory. Among them are implementations of 
[Glow](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/glow.ipynb),
a [VAE](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/vae.py), and
a [Residual Flow](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/residual.ipynb). 
More advanced experiments can be done with the scripts listed in the
[repository about resampled base distributions](https://github.com/VincentStimper/resampled-base-flows),
see its [`experiments`](https://github.com/VincentStimper/resampled-base-flows/tree/master/experiments)
folder.

Below, we consider two simple 2D examples.

### Real NVP applied to a 2D bimodal target distribution

<a href="https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/real_nvp_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In [this notebook](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/real_nvp_colab.ipynb), 
which can directly be opened in 
[Colab](https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/real_nvp_colab.ipynb),
we consider a 2D distribution with two half-moon-shaped modes as a target. We approximate it with a Real NVP model
and obtain the following results.

![2D target distribution and Real NVP model](https://raw.githubusercontent.com/VincentStimper/normalizing-flows/master/figures/real_nvp.png)

Note that there might be a density filament connecting the two modes, which is due to an architectural limitation 
of normalizing flows, especially prominent in Real NVP. You can find out more about it in 
[this paper](https://proceedings.mlr.press/v151/stimper22a).

### Modeling a distribution on a cylinder surface with a neural spline flow

<a href="https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/paper_example_nsf_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

In [another example](https://github.com/VincentStimper/normalizing-flows/blob/master/examples/paper_example_nsf_colab.ipynb),
which is available in [Colab](https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/paper_example_nsf_colab.ipynb)
as well, we apply a Neural Spline Flow model to a distribution defined on a cylinder. The resulting density is visualized below.

![Neural Spline Flow applied to target distribution on a cylinder](https://raw.githubusercontent.com/VincentStimper/normalizing-flows/master/figures/nsf_cylinder_3d.png)

This example is considered in the [paper](https://joss.theoj.org/papers/10.21105/joss.05361) accompanying this repository.

## Support

If you have problems, please read the [package documentation](https://vincentstimper.github.io/normalizing-flows/)
and check out the [examples section](#examples) above. You are also welcome to 
[create issues on GitHub](https://github.com/VincentStimper/normalizing-flows/issues) to get help. Note that it is
worthwhile browsing the existing [open](https://github.com/VincentStimper/normalizing-flows/issues?q=is%3Aopen+is%3Aissue) 
and [closed](https://github.com/VincentStimper/normalizing-flows/issues?q=is%3Aissue+is%3Aclosed) issues, which might
address the problem you are facing.

## Contributing

If you find a bug or have a feature request, please 
[file an issue on GitHub](https://github.com/VincentStimper/normalizing-flows/issues).

You are welcome to contribute to the package by fixing the bug or adding the feature yourself. If you want to 
contribute, please add tests for the code you added or modified and ensure it passes successfully by running `pytest`.
This can be done by simply executing
```
pytest
```
within your local version of the repository. Make sure you code is well documented, and we also encourage contributions
to the existing documentation. Once you finished coding and testing, please 
[create a pull request on GitHub](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

## Used by

The package has been used in several research papers, which are listed below.

> Andrew Campbell, Wenlong Chen, Vincent Stimper, José Miguel Hernández-Lobato, and Yichuan Zhang. 
> [A gradient based strategy for Hamiltonian Monte Carlo hyperparameter optimization](https://proceedings.mlr.press/v139/campbell21a.html). 
> In Proceedings of the 38th International Conference on Machine Learning, pp. 1238–1248. PMLR, 2021.
> 
> [Code available on GitHub.](https://github.com/VincentStimper/hmc-hyperparameter-tuning)

> Vincent Stimper, Bernhard Schölkopf, José Miguel Hernández-Lobato. 
> [Resampling Base Distributions of Normalizing Flows](https://proceedings.mlr.press/v151/stimper22a). 
> In Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151, pp. 4915–4936, 2022.
> 
> [Code available on GitHub.](https://github.com/VincentStimper/resampled-base-flows)

> Laurence I. Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, José Miguel Hernández-Lobato. 
> [Flow Annealed Importance Sampling Bootstrap](https://arxiv.org/abs/2208.01893). 
> arXiv preprint arXiv:2208.01893, 2022.
> 
> [Code available on GitHub.](https://github.com/lollcat/fab-torch)

Moreover, the [`boltzgen`](https://github.com/VincentStimper/boltzmann-generators) package
has been build upon `normflows`.

## Citation

If you use `normflows`, please cite the 
[corresponding paper](https://joss.theoj.org/papers/10.21105/joss.05361) as follows.

> Stimper et al., (2023). normflows: A PyTorch Package for Normalizing Flows. 
> Journal of Open Source Software, 8(86), 5361, https://doi.org/10.21105/joss.05361

**Bibtex**

```
@article{Stimper2023, 
  author = {Vincent Stimper and David Liu and Andrew Campbell and Vincent Berenz and Lukas Ryll and Bernhard Schölkopf and José Miguel Hernández-Lobato}, 
  title = {normflows: A PyTorch Package for Normalizing Flows}, 
  journal = {Journal of Open Source Software}, 
  volume = {8},
  number = {86}, 
  pages = {5361}, 
  publisher = {The Open Journal}, 
  doi = {10.21105/joss.05361}, 
  url = {https://doi.org/10.21105/joss.05361}, 
  year = {2023}
} 
```





