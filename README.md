# Normalizing Flows

## Introduction

This is a PyTorch implementation of several normalizing flow, including a variational autoencoder.


## Implemented Flows

* Planar flow ([Rezende & Mohamed, 2015](https://arxiv.org/abs/1505.05770))
* Radial flow ([Rezende & Mohamed, 2015](https://arxiv.org/abs/1505.05770))
* NICE ([Dinh et al., 2014](https://arxiv.org/abs/1410.8516))
* Real NVP ([Dinh et al., 2016](https://arxiv.org/abs/1605.08803))
* Glow ([Kingma & Dhariwal, 2018](https://arxiv.org/abs/1807.03039))
* Residual flow ([Chen et al., 2019](https://arxiv.org/abs/1906.02735))


## Methods of Installation

The latest version of the package can be installed via pip

```
pip install --upgrade git+https://github.com/VincentStimper/normalizing-flows.git
```

If you want to use a GPU, make sure that pytorch is set up correctly by
by following the instructions at the
[PyTorch website](https://pytorch.org/get-started/locally/).

To run the example notebooks clone the repository first

```
git clone https://github.com/VincentStimper/normalizing-flows.git
```

and then install the dependencies.

```
pip install -r requirements_examples.txt
```



