from typing import Optional

import pyro
import torch
from pyro import distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from torch import nn
from torch.distributions import constraints


class BNN(PyroModule):
    def __init__(
        self, d_in: int, d_out: int, n_hidden: int, d_hidden: Optional[int] = None
    ):
        super().__init__()

        # create BNN
        assert d_out == 1
        modules = create_bnn_modules(
            d_in=d_in, d_out=d_out, n_hidden=n_hidden, d_hidden=d_hidden
        )
        self.bnn = PyroModule[nn.Sequential](*modules)

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        mean = self.bnn(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


class MTBNN(PyroModule):
    def __init__(
        self, d_in: int, d_out: int, n_hidden: int, d_hidden: Optional[int] = None
    ):
        super().__init__()

        ## create BNN
        assert d_out == 1
        modules = create_bnn_modules(
            d_in=d_in, d_out=d_out, n_hidden=n_hidden, d_hidden=d_hidden
        )
        self.bnn = PyroModule[nn.Sequential](*modules)

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        with pyro.plate("tasks", x.shape[0]):
            mean = self.bnn(x)
            with pyro.plate("data", x.shape[1]):
                pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

    def guide(self, x, y=None):
        sigma_loc = pyro.param(
            "sigma_loc", torch.tensor(1.0), constraint=constraints.positive
        )
        pyro.sample("sigma", dist.Normal(sigma_loc, 0.05))
        with pyro.plate("tasks", x.shape[0]):
            cur_guide = AutoDiagonalNormal(model=self)
            latents = cur_guide.sample_latent()


def create_bnn_modules(
    d_in: int, d_out: int, n_hidden: int, d_hidden: Optional[int] = None
):
    is_linear_model = n_hidden == 0
    if is_linear_model:
        assert d_hidden is None
        d_hidden = d_out

    modules = []

    # TODO: note that .to_event(2) is different from using multivariate normals
    # TODO: understand that: https://pyro.ai/examples/tensor_shapes.html

    # input layer
    input_layer = PyroModule[nn.Linear](d_in, d_hidden)
    input_layer.weight = PyroSample(
        dist.Normal(0.0, 1.0).expand([d_hidden, d_in]).to_event(2)
    )
    input_layer.bias = PyroSample(dist.Normal(0.0, 1.0).expand([d_hidden]).to_event(1))
    modules.append(input_layer)
    if is_linear_model:
        return modules
    modules.append(PyroModule[nn.Tanh]())

    # hidden layers
    for _ in range(n_hidden - 1):
        hidden_layer = PyroModule[nn.Linear](d_hidden, d_hidden)
        hidden_layer.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([d_hidden, d_hidden]).to_event(2)
        )
        hidden_layer.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([d_hidden]).to_event(1)
        )
        modules.append(hidden_layer)
        modules.append(PyroModule[nn.Tanh]())

    # output layer
    output_layer = PyroModule[nn.Linear](d_hidden, d_out)
    output_layer.weight = PyroSample(
        dist.Normal(0.0, 1.0).expand([d_out, d_hidden]).to_event(2)
    )
    output_layer.bias = PyroSample(dist.Normal(0.0, 1.0).expand([d_out]).to_event(1))
    modules.append(output_layer)

    return modules
