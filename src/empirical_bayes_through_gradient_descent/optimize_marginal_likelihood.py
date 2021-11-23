import math

import numpy as np
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
from torch.optim.lr_scheduler import ExponentialLR
from empirical_bayes_through_gradient_descent.qmc_sampling import QMCNormal

"""
https://www.notion.so/
Multi-Task-BNN-c08430fc9ee44502b1dc5ce6ec1f50da#222ea3e2cc1b4afb9778c4f3b03c4668
"""


class BaseModel(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n, qmc):
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

        if not qmc:
            self.prior_z = lambda self: dist.Normal(
                loc=self.mu_z, scale=self.sigma_z
            ).to_event(1)
        else:
            self.prior_z = lambda self: dist.Independent(
                QMCNormal(loc=self.mu_z, scale=self.sigma_z), 1
            )
        self.z = PyroSample(self.prior_z)


class GlobalLVM(BaseModel):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n, qmc):
        super().__init__(
            mu_z_init=mu_z_init, sigma_z_init=sigma_z_init, sigma_n=sigma_n, qmc=qmc
        )

    def forward(self, L=None, N=None, y=None):
        assert not ((N is None) and (L is not None))
        assert not ((L is None) and (N is not None))
        assert not ((N is None) and (y is None))
        if N is None:
            L = y.shape[0]
            N = y.shape[1]
        if y is not None:
            assert y.ndim == 3
            assert y.shape[0] == L
            assert y.shape[1] == N

        with pyro.plate("tasks", size=L, dim=-2):
            z = self.z
            with pyro.plate("data", size=N, dim=-1):
                likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
                obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


class LocalLVM(BaseModel):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n, qmc):
        super().__init__(
            mu_z_init=mu_z_init, sigma_z_init=sigma_z_init, sigma_n=sigma_n, qmc=qmc
        )

    def forward(self, L=None, N=None, y=None):
        assert not ((N is None) and (L is not None))
        assert not ((L is None) and (N is not None))
        assert not ((N is None) and (y is None))
        if N is None:
            L = y.shape[0]
            N = y.shape[1]
        if y is not None:
            assert y.ndim == 3
            assert y.shape[0] == L
            assert y.shape[1] == N

        with pyro.plate("tasks", size=L, dim=-2):
            with pyro.plate("data", size=N, dim=-1):
                z = self.z
                likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
                obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


def predict(model, L, N, S):
    predictive = Predictive(
        model=model,
        guide=None,
        num_samples=S,
        parallel=True,
        return_sites=("obs",),
    )
    samples = predictive(L=L, N=N, y=None)
    samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}

    return samples


def marginal_log_likelihood(model, y, S, marginalized):
    L = y.shape[0]
    N = y.shape[1]

    ## obtain vectorized model trace
    vectorized_model = pyro.plate("batch", size=S, dim=-3)(model)
    model_trace = poutine.trace(vectorized_model).get_trace(y=y)

    ## compute log-likelihood for the observation sites
    obs_site = model_trace.nodes["obs"]
    log_prob = obs_site["fn"].log_prob(obs_site["value"])  # evaluate likelihoods
    assert log_prob.shape == (S, L, N)

    ## compute predictive likelihood
    if marginalized:
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # sample dim
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # task dim
        log_prob = torch.sum(log_prob, dim=2, keepdim=True)  # data dim
        assert log_prob.shape == (1, 1, 1)
        log_prob = log_prob.squeeze()
        log_prob = log_prob - L * N * torch.log(torch.tensor(S))
    else:
        log_prob = torch.sum(log_prob, dim=2, keepdim=True)  # data dim
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=True)  # sample dim
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)  # task dim
        assert log_prob.shape == (1, 1, 1)
        log_prob = log_prob.squeeze()
        log_prob = log_prob - L * torch.log(torch.tensor(S))

    return log_prob


def true_marginal_log_likelihood(model, y, marginalized):
    L = y.shape[0]
    N = y.shape[1]

    if marginalized:
        mu = model.mu_z
        sigma = torch.sqrt(model.sigma_z ** 2 + model.sigma_n ** 2)
        dist = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob = dist.log_prob(y.squeeze()).sum(-1)  # TODO: use event-dim
    else:
        mu = model.mu_z * torch.ones((N,))
        Sigma = model.sigma_z ** 2 * torch.ones((N, N))
        Sigma = Sigma + model.sigma_n ** 2 * torch.eye(N)
        dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
        log_prob = dist.log_prob(y.squeeze())

    assert log_prob.shape == (L,)
    log_prob = log_prob.sum()

    return log_prob


def plot_samples(samples, ax, label):
    ax.scatter(x=samples, y=torch.zeros(samples.shape), marker="x")
    sns.kdeplot(x=samples, ax=ax, label=label)


def generate_data(L, N, sigma_n, mu_z, sigma_z, correlated=True):
    if not correlated:
        raise NotImplementedError

    y = torch.zeros(L, N, 1)
    for l in range(L):
        z = torch.distributions.Normal(loc=mu_z, scale=sigma_z).sample()
        for n in range(N):
            y[l, n, 0] = torch.distributions.Normal(loc=z, scale=sigma_n).sample()
    return y


# def compute_optimal_prior_parameters(model_type, sigma_n_true, mu_z_true, sigma_z_true):
#     # TODO: check this!
#     # TODO: compute this from the data not from the true parameters!

#     mu_z_opt = mu_z_true
#     if model_type == "local_lvm":
#         sigma_z_opt = math.sqrt(sigma_z_true ** 2 - sigma_n_true ** 2)
#     else:
#         assert model_type == "global_lvm"
#         sigma_z_opt = sigma_z_true

#     return mu_z_opt, sigma_z_opt


def compute_optimal_prior_parameters(model_type, sigma_n_true, y):
    from sklearn.covariance import EmpiricalCovariance

    L = y.shape[0]
    N = y.shape[1]
    # assert N == 2

    mu_z_opt = y.mean()
    Sigma = EmpiricalCovariance().fit(y.squeeze(-1))

    if model_type == "local_lvm":
        sigma_z_opt = np.sqrt(
            (Sigma.covariance_[0, 0] + Sigma.covariance_[1, 1]) / 2 - sigma_n_true ** 2
        )
    else:
        assert model_type == "global_lvm"
        sigma_z_opt = np.sqrt(Sigma.covariance_[0, 1])

    return mu_z_opt, sigma_z_opt


def main():
    ### settings
    pyro.set_rng_seed(123)
    n_epochs = 10000
    initial_lr = 1e-2
    final_lr = 1e-2
    S_marg_ll_est = 2 ** 10
    S_plot = 2 ** 7
    S_grad_est = 2 ** 0  # test this for 1, 5, 10, 100 (-> gets noisier with S_grad_est)
    ## data
    L = 10
    N = 100
    mu_z_true = 1.0
    sigma_z_true = 0.1
    sigma_n_true = 0.01
    y = generate_data(
        L=L, N=N, sigma_n=sigma_n_true, mu_z=mu_z_true, sigma_z=sigma_z_true
    )
    ## model
    # type
    # model_type = "local_lvm"
    model_type = "global_lvm"
    qmc = False 
    # mu_z_opt, sigma_z_opt = compute_optimal_prior_parameters(
    #     model_type=model_type,
    #     sigma_n_true=sigma_n_true,
    #     mu_z_true=mu_z_true,
    #     sigma_z_true=sigma_z_true,
    # )
    mu_z_opt, sigma_z_opt = compute_optimal_prior_parameters(
        model_type=model_type, sigma_n_true=sigma_n_true, y=y
    )
    mu_z_init = -1.0
    sigma_z_init = 1.0
    sigma_n_model = sigma_n_true
    marginalized_ll = True if model_type == "local_lvm" else False

    ### generate model
    assert sigma_n_model == sigma_n_true  # the optimal solutions are based on this
    model = LocalLVM if model_type == "local_lvm" else GlobalLVM
    model = model(
        mu_z_init=mu_z_init,
        sigma_z_init=sigma_z_init,
        sigma_n=sigma_n_model,
        qmc=qmc,
    )

    ### compute predictive distribution before training
    samples_init = predict(model=model, L=L, N=N, S=S_plot)
    samples_init = samples_init["obs"].squeeze().flatten()

    ### compute marginal likelihood before training
    N_S = 10
    marg_ll_sampled_init = torch.zeros(N_S)
    for i in range(N_S):
        marg_ll_sampled_init[i] = marginal_log_likelihood(
            model=model, y=y, S=S_marg_ll_est, marginalized=marginalized_ll
        )
    marg_ll_samp_mean_init = marg_ll_sampled_init.mean()
    marg_ll_samp_std_init = marg_ll_sampled_init.std()
    try:
        marg_ll_true_init = true_marginal_log_likelihood(
            model=model, y=y, marginalized=marginalized_ll
        )
    except ValueError:
        marg_ll_true_init = None

    ### optimize prior
    print(f"mu_z_opt  = {mu_z_opt:.4f}")
    print(f"std_z_opt = {sigma_z_opt:.4f}")
    losses = []
    mu_zs = []
    sigma_zs = []
    optim = torch.optim.Adam(lr=initial_lr, params=model.parameters())
    gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
    lr_decay = gamma ** (1 / n_epochs)
    lr_scheduler = ExponentialLR(optimizer=optim, gamma=lr_decay)
    for epoch in range(n_epochs):
        optim.zero_grad()
        loss = -marginal_log_likelihood(
            model=model, y=y, S=S_grad_est, marginalized=marginalized_ll
        )

        # log
        losses.append(loss.item())
        mu_zs.append(model.mu_z.item())
        sigma_zs.append(model.sigma_z.item())

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            # print_pyro_parameters()
            print(
                f"epoch = {epoch:04d}"
                f" | loss = {loss.item():.4f}"
                f" | mu_z = {model.mu_z.item():.4f}"
                f" | sigma_z = {model.sigma_z.item():.4f}"
            )

        loss.backward()
        optim.step()
        lr_scheduler.step()

    ### plot predictive distribution (trained)
    samples_trained = predict(model=model, L=L, N=N, S=S_plot)
    samples_trained = samples_trained["obs"].squeeze().flatten()

    ### compute marginal likelihood (trained)
    N_S = 10
    marg_ll_sampled_trained = torch.zeros(N_S)
    for i in range(N_S):
        marg_ll_sampled_trained[i] = marginal_log_likelihood(
            model=model, y=y, S=S_marg_ll_est, marginalized=marginalized_ll
        )
    marg_ll_samp_mean_trained = marg_ll_sampled_trained.mean()
    marg_ll_samp_std_trained = marg_ll_sampled_trained.std()
    try:
        marg_ll_true_trained = true_marginal_log_likelihood(
            model=model, y=y, marginalized=marginalized_ll
        )
    except ValueError:
        marg_ll_true_trained = None

    ### print
    print("*" * 50)
    print(f"data mean                  = {y.mean():+.4f}")
    print(f"data std                   = {y.std():+.4f}")
    print(f"obs std                    = {model.sigma_n:+.4f}")
    print("*" * 20)
    print(f"prior mean (init)          = {mu_z_init:+.4f}")
    print(f"prior std (init)           = {sigma_z_init:+.4f}")
    print(
        f"marg ll (init, sampled)    = {marg_ll_samp_mean_init:+.4f} "
        f"+- {marg_ll_samp_std_init:.4f}"
    )
    if marg_ll_true_init is not None:
        print(f"marg ll (init, true)       = {marg_ll_true_init:+.4f}")
    else:
        print(f"marg ll (init, true)       = computation error")
    print(f"predictive mean (init)     = {samples_init.mean():+.4f}")
    print(f"predictive std (init)      = {samples_init.std():+.4f}")
    print("*" * 20)
    print(f"prior mean (trained)       = {model.mu_z.item():+.4f}")
    print(f"prior std (trained)        = {model.sigma_z.item():+.4f}")
    print(
        f"marg ll (trained, sampled) = {marg_ll_samp_mean_trained:+.4f} "
        f"+- {marg_ll_samp_std_trained:.4f}"
    )
    if marg_ll_true_trained is not None:
        print(f"marg ll (trained, true)    = {marg_ll_true_trained:+.4f}")
    else:
        print(f"marg ll (trained, true)    = computation error")
    print(f"predictive mean (trained)  = {samples_trained.mean():+.4f}")
    print(f"predictive std (trained)   = {samples_trained.std():+.4f}")
    print("*" * 20)
    print(f"prior mean (opt)           = {mu_z_opt:+.4f}")
    print(f"prior std (opt)            = {sigma_z_opt:+.4f}")
    print("*" * 50)

    ### plot prediction
    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(18, 10))
    ax = axes[0, 0]
    plot_samples(y.reshape(-1), ax=ax, label="data")
    plot_samples(samples_init, ax=ax, label="predictions (init)")
    plot_samples(samples_trained, ax=ax, label="predictions (trained)")
    ax.set_title("Data and Marginal Predictions")
    ax.set_xlabel("y_i")
    ax.set_ylabel("p(y_i)")
    ax.grid()
    ax.legend()

    ax = axes[0, 1]
    ax.plot(np.arange(len(losses)), losses)
    ax.set_title("Learning curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid()
    ax.legend()

    ax = axes[0, 2]
    ax.scatter(x=mu_zs, y=sigma_zs, c=np.arange(n_epochs) + 1)
    ax.axvline(mu_z_opt, ls="--")
    ax.axhline(sigma_z_opt, ls="--")
    ax.set_xlabel("mu_z")
    ax.set_ylabel("sigma_z")
    ax.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
