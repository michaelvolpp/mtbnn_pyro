import math

import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from mtutils.mtutils import print_pyro_parameters
from numpy import dtype
from pyro import distributions as dist
from pyro import poutine
from pyro.infer import EmpiricalMarginal, Importance, Predictive
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch.distributions import constraints

"""
https://www.notion.so/
Multi-Task-BNN-c08430fc9ee44502b1dc5ce6ec1f50da#222ea3e2cc1b4afb9778c4f3b03c4668
"""


class BaseModel(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n):
        super().__init__()

        # likelihood
        self.sigma_n = sigma_n

        # prior
        self.mu_z = PyroParam(
            init_value=torch.tensor([mu_z_init]),
            constraint=constraints.real,
        )
        self.sigma_z = PyroParam(
            init_value=torch.tensor([sigma_z_init]),
            constraint=constraints.positive,
        )
        self.prior_z = lambda self: dist.Normal(
            loc=self.mu_z, scale=self.sigma_z
        ).to_event(1)
        self.z = PyroSample(self.prior_z)


class MultiTaskModel(BaseModel):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n):
        super().__init__(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n,
        )

    def forward(self, N=None, y=None):
        if N is None:
            assert y is not None
            N = y.shape[0]
        if y is not None:
            assert y.ndim == 2
            assert y.shape[0] == N

        z = self.z
        with pyro.plate("data", size=N, dim=-1):
            likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
            obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


class SingleTaskModel(BaseModel):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n):
        super().__init__(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n,
        )

    def forward(self, N=None, y=None):
        if N is None:
            assert y is not None
            N = y.shape[0]
        if y is not None:
            assert y.ndim == 2
            assert y.shape[0] == N

        with pyro.plate("data", size=N, dim=-1):
            z = self.z
            likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
            obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


def predict(model, N, S):
    predictive = Predictive(
        model=model,
        guide=None,
        num_samples=S,
        parallel=True,
        return_sites=("obs",),
    )
    samples = predictive(N=N, y=None)
    samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}

    return samples


def marginal_log_likelihood(model, y, S):
    N = y.shape[0]

    ## obtain vectorized model trace
    vectorized_model = pyro.plate("batch", size=S, dim=-2)(model)
    model_trace = poutine.trace(vectorized_model).get_trace(y=y)

    ## compute log-likelihood for the observation sites
    obs_site = model_trace.nodes["obs"]
    log_prob = obs_site["fn"].log_prob(obs_site["value"])  # evaluate likelihoods
    assert log_prob.shape == (S, N)

    ## compute predictive likelihood
    # TODO: unify the following
    if isinstance(model, SingleTaskModel):
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # sample dim
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # data dim
        assert log_prob.shape == (1, 1)
        log_prob = log_prob.squeeze()
        log_prob = log_prob - N * torch.log(torch.tensor(S))
    else:
        assert isinstance(model, MultiTaskModel)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # data dim
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # sample dim
        assert log_prob.shape == (1, 1)
        log_prob = log_prob.squeeze()
        log_prob = log_prob - torch.log(torch.tensor(S))

    return log_prob


def true_marginal_log_likelihood(model, y):
    N = y.shape[0]

    # TODO: unify the following
    if isinstance(model, SingleTaskModel):
        mu = model.mu_z
        sigma = torch.sqrt(model.sigma_z ** 2 + model.sigma_n ** 2)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob = dist.log_prob(y.squeeze()).sum()
    else:
        assert isinstance(model, MultiTaskModel)
        mu = model.mu_z * torch.ones((N,))
        Sigma = model.sigma_z ** 2 * torch.ones((N, N))
        Sigma = Sigma + model.sigma_n ** 2 * torch.eye(N)
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
        log_prob = dist.log_prob(y.squeeze())

    return log_prob


def plot_samples(samples, ax, label):
    ax.scatter(x=samples, y=torch.zeros(samples.shape), marker="x")
    sns.kdeplot(x=samples, ax=ax, label=label)


def main():
    ### settings
    pyro.set_rng_seed(123)
    S_marg_ll_est = 10000
    S_grad_est = 10000
    S_plot = 1000
    ## data
    mu_y = -1.0
    sigma_y = 0.5
    N = 1000
    y = torch.randn((N, 1)) * sigma_y + mu_y
    ## model
    # type
    # model_type = "single_task"
    model_type = "multi_task"
    sigma_n = 0.1

    ### compute optimal solutions for the prior
    assert sigma_n < sigma_y
    mu_z_opt = mu_y
    if model_type == "single_task":
        sigma_z_opt = math.sqrt(sigma_y ** 2 - sigma_n ** 2)
    else:
        sigma_z_opt = math.sqrt(1 / N * (sigma_y ** 2 - sigma_n ** 2))

    ### initial prior
    # mu_z_init = 1.0
    # sigma_z_init = 1.0
    mu_z_init = mu_z_opt
    sigma_z_init = sigma_z_opt

    ### generate model
    model = SingleTaskModel if model_type == "single_task" else MultiTaskModel
    model = model(mu_z_init=mu_z_init, sigma_z_init=sigma_z_init, sigma_n=sigma_n)

    ### compute predictive distribution before training
    samples = predict(model=model, N=N, S=S_plot)
    samples = samples["obs"].squeeze().flatten()

    ### compute marginal likelihood before training
    # N_S = 10
    # marg_ll_sampled = torch.zeros(N_S)
    # for i in range(N_S):
    #     marg_ll_sampled[i] = marginal_log_likelihood(model=model, y=y, S=S_marg_ll_est)
    # marg_ll_samp_mean = marg_ll_sampled.mean()
    # marg_ll_samp_std = marg_ll_sampled.std()
    # marg_ll_true = true_marginal_log_likelihood(model=model, y=y)

    ### print
    print("*" * 100)
    print(f"data mean         = {y.mean():+.4f}")
    print(f"data std          = {y.std():+.4f}")
    print(f"cos(pi/4) * data std = {math.cos(math.pi/4)*y.std():.4f}")
    print(f"prior mean        = {model.mu_z.item():+.4f}")
    print(f"prior std         = {model.sigma_z.item():+.4f}")
    print(f"obs std           = {model.sigma_n:+.4f}")
    # print(f"marg ll (sampled) = {marg_ll_samp_mean:+.4f} +- {marg_ll_samp_std:.4f}")
    # print(f"marg ll (true)    = {marg_ll_true:+.4f}")
    print(f"predictive mean   = {samples.mean():+.4f}")
    print(f"predictive std    = {samples.std():+.4f}")
    print("*" * 100)

    ### plot prior prediction (untrained)
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    ax = axes[0, 0]
    plot_samples(y.squeeze(-1), ax=ax, label="data")
    plot_samples(samples, ax=ax, label="before training")

    # ### optimize prior
    # print(f"mu_z_opt  = {mu_z_opt:.4f}")
    # print(f"std_z_opt = {sigma_z_opt:.4f}")
    # optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    # n_epochs = 10000
    # for epoch in range(n_epochs):
    #     optimizer.zero_grad()
    #     loss = -marginal_log_likelihood(model=model, y=y, S=S_grad_est)

    #     if epoch % 100 == 0 or epoch == n_epochs - 1:
    #         # print_pyro_parameters()
    #         print(
    #             f"epoch = {epoch:04d}"
    #             f" | loss = {loss.item():.4f}"
    #             f" | mu_z = {model.mu_z.item():.4f}"
    #             f" | sigma_z = {model.sigma_z.item():.4f}"
    #         )

    #     loss.backward()
    #     optimizer.step()
    # print(f"mu_z_opt  = {mu_z_opt:.4f}")
    # print(f"std_z_opt = {sigma_z_opt:.4f}")

    # ### plot prior prediction (trained)
    # samples = predict(model=model, N=N, S=S_plot)
    # samples = samples["obs"].squeeze().flatten()
    # plot_samples(samples, ax=ax, label="after training")
    # print(f"predictive mean after training = {samples.mean():.4f}")
    # print(f"predictive std after training  = {samples.std():.4f}")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
