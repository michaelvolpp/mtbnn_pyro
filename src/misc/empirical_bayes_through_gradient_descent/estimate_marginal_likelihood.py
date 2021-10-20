import pyro
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyro import distributions as dist
from pyro import poutine
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch.distributions import constraints
from tqdm import tqdm


class Model(PyroModule):
    def __init__(self, init_mu, init_sigma, sigma_n):
        super().__init__()

        # prior
        self.mu = PyroParam(
            init_value=torch.tensor([init_mu]),
            constraint=constraints.real,
        )
        self.sigma = PyroParam(
            init_value=torch.tensor([init_sigma]),
            constraint=constraints.positive,
        )
        self.prior = lambda self: dist.Normal(loc=self.mu, scale=self.sigma).to_event(1)
        self.z = PyroSample(self.prior)
        self.sigma_n = sigma_n

    def forward(self, dataset_size=None, y=None):
        """
        p(y, z) = p(y|z) * p(z)
        p(z) = N(z | mu, sigma^2)
        p(y|z) = N(y | z, sigma_n^2)
        -> marginal mean: mu
        -> marginal std: sqrt(sigma^2 + sigma_n^2)
        """
        if dataset_size is None:
            assert y is not None
            dataset_size = y.shape[0]
        if y is not None:
            assert y.ndim == 2
            assert y.shape[0] == dataset_size

        with pyro.plate("data", size=dataset_size, dim=-1):
            z = self.z
            likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
            obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


def marginal_log_likelihood(model, y, S):
    L = y.shape[0]

    # obtain vectorized model trace
    vectorized_model = pyro.plate("batch", size=S, dim=-2)(model)
    model_trace = poutine.trace(vectorized_model).get_trace(y=y)

    # compute log-likelihood for the observation sites
    obs_site = model_trace.nodes["obs"]
    log_prob = obs_site["fn"].log_prob(obs_site["value"])  # reduces event-dims
    assert log_prob.shape == (S, L)

    # compute predictive likelihood
    log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # reduce sample dim
    log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # sum task dim
    assert log_prob.shape == (1, 1)
    log_prob = log_prob.squeeze()
    log_prob = log_prob - L * torch.log(torch.tensor(S))

    # normalize w.r.t. number of datapoints
    log_prob = log_prob / L

    return log_prob


def true_marginal_log_likelihood(model, y, sigma_n):
    L = y.shape[0]
    mu = model.mu
    sigma = torch.sqrt(model.sigma ** 2 + sigma_n ** 2)
    dist = torch.distributions.Normal(loc=mu, scale=sigma)

    log_prob = dist.log_prob(y.squeeze()).sum() / L

    return log_prob


def main():
    pyro.set_rng_seed(123)

    # data
    true_mean = -1.0
    true_std = 1.0
    sigma_n = 1.0
    # L = 1
    # y = torch.randn((L, 1)) * true_std + true_mean
    # y = y + torch.randn((L, 1)) * sigma_n
    # print(f"true mean = {y.mean():.4f}")
    # print(f"true std  = {y.std():.4f}")
    # print(f"y         = {y}")
    y = -1.0 * torch.ones((1, 1))

    # model
    init_mu = 1.0
    init_sigma = 1.0
    model = Model(init_mu=init_mu, init_sigma=init_sigma, sigma_n=sigma_n)

    # estimate marginal likelihood
    S_list = np.logspace(0, 8, num=20, dtype=int)
    marg_ll_ests = []
    relative_errs = []
    marg_ll_true = (
        true_marginal_log_likelihood(model=model, y=y, sigma_n=sigma_n).detach().numpy()
    )
    for S in tqdm(S_list):
        cur_marg_ll_est = (
            marginal_log_likelihood(model=model, y=y, S=S).detach().numpy()
        )
        marg_ll_ests.append(cur_marg_ll_est)
        relative_errs.append(np.abs((cur_marg_ll_est - marg_ll_true) / marg_ll_true))

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), squeeze=False)
    ax = axes[0, 0]
    ax.plot(S_list, marg_ll_ests, label="estimated")
    ax.axhline(y=marg_ll_true, label="true", color="r")
    ax.set_xscale("log")
    ax.set_title("estimated vs. true marginal log-likelihood")
    ax.grid()
    ax.legend()

    ax = axes[0, 1]
    ax.plot(S_list, marg_ll_ests, label="estimated")
    ax.axhline(y=marg_ll_true, label="true", color="r")
    ax.set_xscale("log")
    ax.set_ylim(marg_ll_true * 0.75, marg_ll_true * 1.25)
    ax.set_title("estimated vs. true marginal log-likelihood\n(rescaled y-axis)")
    ax.grid()
    ax.legend()

    ax = axes[0, 2]
    ax.plot(S_list, relative_errs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("relative error of marginal-log-likelihood estimator")
    ax.grid()
    ax.legend()

    fig.suptitle(f"{sigma_n = }")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
