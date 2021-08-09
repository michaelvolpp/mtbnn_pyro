from numpy import ndarray
import pyro
from pyro import distributions as dist
from torch import distributions
import torch


def model(d_z, n_data):
    with pyro.plate("data", n_data):
        prior_loc = torch.zeros(n_data + 1, d_z)
        prior_scale = torch.ones(n_data + 1, d_z)
        prior = dist.Normal(prior_loc, prior_scale).to_event(1)
        z = pyro.sample("z", prior)
    return z


if __name__ == "__main__":
    d_z = 2
    n_data = 7
    z = model(d_z=d_z, n_data=n_data)
    print(z.shape)
    print(z)
