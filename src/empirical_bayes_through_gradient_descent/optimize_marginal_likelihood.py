import math

import numpy as np
import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.autograd import grad
from mtutils.mtutils import print_pyro_parameters
from numpy import dtype
from pyro import distributions as dist
from pyro import poutine
from pyro.infer import Predictive
from pyro.nn import PyroModule, PyroParam, PyroSample
from torch.distributions import constraints
from torch.optim.lr_scheduler import ExponentialLR
from pyro.infer.renyi_elbo import RenyiELBO
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.autoguide import AutoNormal

from empirical_bayes_through_gradient_descent.qmc_sampling import QMCNormal

"""
https://www.notion.so/
Multi-Task-BNN-c08430fc9ee44502b1dc5ce6ec1f50da#222ea3e2cc1b4afb9778c4f3b03c4668
"""

# - due to set_attr behaviour, guide and model share mu_z, sigma_z => unintended?
# - check that importance weight is zero for qmz
# - understand difference in loss between iwmc and vi (log <-> integral) -> what is then the difference to gordon, they interchange log and integral

allowed_model_types = ["local_lvm", "global_lvm"]
allowed_guide_types = ["prior", "qmc_prior", "true_posterior", "approximate_posterior"]
allowed_log_marginal_likelihood_estimators = ["standard_elbo", "iwae_elbo"]


def check_consistency(L, N, y):
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

    return L, N


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

    @PyroSample
    def z(self):
        return dist.Normal(loc=self.mu_z, scale=self.sigma_z).to_event(1)


class GlobalLVM(BaseModel):
    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)

        with pyro.plate("tasks", size=L, dim=-2):
            z = self.z
            with pyro.plate("data", size=N, dim=-1):
                likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
                obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


class LocalLVM(BaseModel):
    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)

        with pyro.plate("tasks", size=L, dim=-2):
            with pyro.plate("data", size=N, dim=-1):
                z = self.z
                likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
                obs = pyro.sample("obs", fn=likelihood, obs=y)

        return obs


class QMCPriorGlobalLVM(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init):
        super().__init__()

        # use the same attribute names as BaseModel, s.t. the guide parameters are
        # shared with the model

        # prior
        self.mu_z = PyroParam(
            init_value=torch.tensor([mu_z_init]),
            constraint=constraints.real,
        )
        self.sigma_z = PyroParam(
            init_value=torch.tensor([sigma_z_init]),
            constraint=constraints.positive,
        )

    @PyroSample
    def z(self):
        return dist.Independent(QMCNormal(loc=self.mu_z, scale=self.sigma_z), 1)

    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)

        with pyro.plate("tasks", size=L, dim=-2):
            self.z


class QMCPriorLocalLVM(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init):
        super().__init__()

        # use the same attribute names as BaseModel, s.t. the guide parameters are
        # shared with the model

        # prior
        self.mu_z = PyroParam(
            init_value=torch.tensor([mu_z_init]),
            constraint=constraints.real,
        )
        self.sigma_z = PyroParam(
            init_value=torch.tensor([sigma_z_init]),
            constraint=constraints.positive,
        )

    @PyroSample
    def z(self):
        return dist.Independent(QMCNormal(loc=self.mu_z, scale=self.sigma_z), 1)

    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)

        with pyro.plate("tasks", size=L, dim=-2):
            with pyro.plate("data", size=N, dim=-1):
                self.z


class TruePosteriorGlobalLVM(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n, y):
        super().__init__()

        # use the same attribute names as BaseModel, s.t. the guide parameters are
        # shared with the model

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

        # data
        self.y = y

    @PyroSample
    def z(self):
        N = self.y.shape[1]
        var_post = (self.sigma_n ** 2 * self.sigma_z ** 2) / (
            self.sigma_n ** 2 + N * self.sigma_z ** 2
        )
        mu_post = var_post * (
            1 / self.sigma_n ** 2 * torch.sum(self.y, dim=1, keepdims=True)
            + 1 / self.sigma_z ** 2 * self.mu_z
        )
        return dist.Normal(loc=mu_post, scale=torch.sqrt(var_post)).to_event(1)

    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)
        assert L == self.y.shape[0]
        assert N == self.y.shape[1]

        with pyro.plate("tasks", size=L, dim=-2):
            z = self.z


class TruePosteriorLocalLVM(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n, y):
        super().__init__()

        # use the same attribute names as BaseModel, s.t. the guide parameters are
        # shared with the model

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

        # data
        self.y = y

    @PyroSample
    def z(self):
        N = self.y.shape[1]
        var_post = (self.sigma_n ** 2 * self.sigma_z ** 2) / (
            self.sigma_n ** 2 + self.sigma_z ** 2
        )
        mu_post = var_post * (
            1 / self.sigma_n ** 2 * self.y + 1 / self.sigma_z ** 2 * self.mu_z
        )
        return dist.Normal(loc=mu_post, scale=torch.sqrt(var_post)).to_event(1)

    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)
        assert L == self.y.shape[0]
        assert N == self.y.shape[1]

        with pyro.plate("tasks", size=L, dim=-2):
            with pyro.plate("data", size=N, dim=-1):
                z = self.z


def predict(model, guide, L, N, S):
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=S,
        parallel=True,
        return_sites=("obs", "z"),
    )
    samples = predictive(L=L, N=N, y=None)
    samples = {k: v.detach().cpu().numpy() for k, v in samples.items()}

    return samples


def log_marginal_likelihood(model, guide, y, S, estimator_type):
    # TODO: use RenyiELBO
    # ## validate loss
    # if estimator_type == "iwae_elbo":
    #     elbo = RenyiELBO(alpha=0.0, vectorize_particles=True, num_particles=S)
    # elif estimator_type == "standard_elbo":
    #     elbo = Trace_ELBO(vectorize_particles=True, num_particles=S)
    # else:
    #     raise NotImplementedError
    # true_elbo = elbo.loss(model=model, guide=guide, y=y)
    true_elbo = None

    assert estimator_type in allowed_log_marginal_likelihood_estimators
    L = y.shape[0]
    N = y.shape[1]

    ## sample hidden variables from guide
    # cf. https://docs.pyro.ai/en/1.7.0/_modules/pyro/infer/importance.html#Importance
    # vectorize guide
    vectorized_guide = pyro.plate("batch", size=S, dim=-3)(guide)
    # hide observed samples
    #  -> this is only necessary if we set guide = model, as then the guide also samples
    #     observed sites
    guide_trace = poutine.block(vectorized_guide, hide_types=["observe"])
    # trace guide
    guide_trace = poutine.trace(guide_trace)
    guide_trace = guide_trace.get_trace(y=y)

    ## replay model with hidden samples from guide
    vectorized_model = pyro.plate("batch", size=S, dim=-3)(model)
    replayed_model = poutine.replay(vectorized_model, trace=guide_trace)
    model_trace = poutine.trace(replayed_model).get_trace(y=y)

    ## compute log_probs
    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()

    if isinstance(model, LocalLVM):
        ## sanity checks
        assert (
            model_trace.nodes["z"]["value"] == guide_trace.nodes["z"]["value"]
        ).all()
        assert model_trace.nodes["obs"]["value"].shape == (L, N, 1)
        assert model_trace.nodes["obs"]["fn"].base_dist.loc.shape == (S, L, N, 1)
        assert model_trace.nodes["obs"]["fn"].base_dist.scale.shape == (S, L, N, 1)
        assert model_trace.nodes["obs"]["log_prob"].shape == (S, L, N)
        assert model_trace.nodes["z"]["value"].shape == (S, L, N, 1)
        assert model_trace.nodes["z"]["log_prob"].shape == (S, L, N)
        assert guide_trace.nodes["z"]["value"].shape == (S, L, N, 1)
        assert guide_trace.nodes["z"]["log_prob"].shape == (S, L, N)

        ## compute the log importance weights for each task and sample
        log_prob_lhd = model_trace.nodes["obs"]["log_prob"]
        log_prob_prior = model_trace.nodes["z"]["log_prob"]
        log_prob_guide = guide_trace.nodes["z"]["log_prob"]
        assert (
            log_prob_lhd.shape
            == log_prob_prior.shape
            == log_prob_guide.shape
            == (S, L, N)
        )
        log_iw = log_prob_lhd + log_prob_prior - log_prob_guide
        assert log_iw.shape == (S, L, N)

        ## compute log marginal likelihood
        if estimator_type == "iwae_elbo":
            log_marg_lhd = torch.logsumexp(log_iw, dim=0, keepdim=True)  # sample dim
            log_marg_lhd = torch.sum(log_marg_lhd, dim=2, keepdim=True)  # data dim
            log_marg_lhd = torch.sum(log_marg_lhd, dim=1, keepdim=True)  # task dim
            assert log_marg_lhd.shape == (1, 1, 1)
            log_marg_lhd = log_marg_lhd.squeeze()
            log_marg_lhd = log_marg_lhd - L * N * torch.log(torch.tensor(S))
        elif estimator_type == "standard_elbo":
            log_marg_lhd = torch.sum(log_iw) / S
        else:
            raise NotImplementedError
    else:
        assert isinstance(model, GlobalLVM)

        ## sanity checks
        assert (
            model_trace.nodes["z"]["value"] == guide_trace.nodes["z"]["value"]
        ).all()
        assert model_trace.nodes["obs"]["value"].shape == (L, N, 1)
        assert model_trace.nodes["obs"]["fn"].base_dist.loc.shape == (S, L, N, 1)
        assert model_trace.nodes["obs"]["fn"].base_dist.scale.shape == (S, L, N, 1)
        assert model_trace.nodes["obs"]["log_prob"].shape == (S, L, N)
        assert model_trace.nodes["z"]["value"].shape == (S, L, 1, 1)
        assert model_trace.nodes["z"]["log_prob"].shape == (S, L, 1)
        assert guide_trace.nodes["z"]["value"].shape == (S, L, 1, 1)
        assert guide_trace.nodes["z"]["log_prob"].shape == (S, L, 1)

        ## compute the log importance weights for each task and sample
        log_prob_lhd = torch.sum(
            model_trace.nodes["obs"]["log_prob"], dim=2, keepdim=True
        )
        log_prob_prior = model_trace.nodes["z"]["log_prob"]
        log_prob_guide = guide_trace.nodes["z"]["log_prob"]
        assert (
            log_prob_lhd.shape
            == log_prob_prior.shape
            == log_prob_guide.shape
            == (S, L, 1)
        )
        log_iw = log_prob_lhd + log_prob_prior - log_prob_guide
        assert log_iw.shape == (S, L, 1)

        ## compute log marginal likelihood
        if estimator_type == "iwae_elbo":
            log_marg_lhd = torch.logsumexp(log_iw, dim=0, keepdim=True)  # sample dim
            log_marg_lhd = torch.sum(log_marg_lhd, dim=1, keepdim=True)  # task dim
            assert log_marg_lhd.shape == (1, 1, 1)
            log_marg_lhd = log_marg_lhd.squeeze()
            log_marg_lhd = log_marg_lhd - L * torch.log(torch.tensor(S))
        elif estimator_type == "standard_elbo":
            log_marg_lhd = torch.sum(log_iw) / S
        else:
            raise NotImplementedError

    return log_marg_lhd, log_iw, true_elbo


def true_log_marginal_likelihood(model, y):
    L = y.shape[0]
    N = y.shape[1]

    if isinstance(model, LocalLVM):
        mu = model.mu_z
        sigma = torch.sqrt(model.sigma_z ** 2 + model.sigma_n ** 2)
        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        log_prob = normal.log_prob(y.squeeze(-1)).sum(-1)  # TODO: use event-dim
    else:
        assert isinstance(model, GlobalLVM)
        mu = model.mu_z * torch.ones((N,))
        Sigma = model.sigma_z ** 2 * torch.ones((N, N))
        Sigma = Sigma + model.sigma_n ** 2 * torch.eye(N)
        normal = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
        log_prob = normal.log_prob(y.squeeze(-1))

    assert log_prob.shape == (L,)
    log_prob = log_prob.sum()

    return log_prob


def plot_kde(samples, ax, color, linestyle, label, alpha=1.0):
    sns.kdeplot(
        x=samples, ax=ax, label=label, color=color, linestyle=linestyle, alpha=alpha
    )


def plot_functions(samples, ax, color, linestyle, label, alpha=1.0):
    assert samples.ndim == 1
    ax.plot(
        np.arange(samples.shape[0]),
        samples,
        label=label,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
    )


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
    # TODO: solution for global LVM is not accurate
    from sklearn.covariance import EmpiricalCovariance

    L = y.shape[0]
    N = y.shape[1]

    mu_emp = y.mean(dim=0)
    Sigma_emp = EmpiricalCovariance().fit(y.squeeze(-1))

    mu_z_opt = mu_emp.mean()
    if model_type == "local_lvm":
        var_opt = 1 / N * np.trace(Sigma_emp.covariance_) - sigma_n_true ** 2
        # var_opt2 = ((y - mu_z_opt) ** 2).mean() - sigma_n_true ** 2
        # assert np.isclose(var_opt, var_opt2.item())
        sigma_z_opt = np.sqrt(var_opt)
    else:
        assert model_type == "global_lvm"
        sigma_z_opt = np.sqrt(Sigma_emp.covariance_[0, 1])  # TODO: not accurate for N>2

    return mu_z_opt, sigma_z_opt


def main():
    ### settings
    pyro.set_rng_seed(123)
    optimize = False
    n_epochs = 1000
    initial_lr = 1e-2
    final_lr = 1e-2
    S_plot = 10
    L_plot = 3
    S = 2 ** 0  # test this for 1, 5, 10, 100 (-> gets noisier with S)
    ## data
    L = 10
    N = 50
    mu_z_true = 1.0
    sigma_z_true = 0.5
    sigma_n_true = 0.01
    y = generate_data(
        L=L, N=N, sigma_n=sigma_n_true, mu_z=mu_z_true, sigma_z=sigma_z_true
    )
    ## model
    model_type = "local_lvm"
    # model_type = "global_lvm"
    mu_z_opt, sigma_z_opt = compute_optimal_prior_parameters(
        model_type=model_type, sigma_n_true=sigma_n_true, y=y
    )
    # mu_z_init = -1.0
    # sigma_z_init = 1.0
    mu_z_init = mu_z_true
    sigma_z_init = sigma_z_true
    sigma_n_model = sigma_n_true
    ## guide
    guide_type = "prior"
    # guide_type = "qmc_prior"  # understand scrambling
    # guide_type = "true_posterior"
    # guide_type = "approximate_posterior"
    ## log marginal likelihood estimator
    lmlhd_estimator_type = "iwae_elbo"
    # lmlhd_estimator_type = "standard_elbo"

    assert model_type in allowed_model_types
    assert guide_type in allowed_guide_types

    ### generate model
    assert sigma_n_model == sigma_n_true  # the optimal solutions are based on this
    if model_type == "local_lvm":
        model = LocalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n_model,
        )
    elif model_type == "global_lvm":
        model = GlobalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n_model,
        )

    ### generate true posterior
    if model_type == "local_lvm":
        true_posterior = TruePosteriorLocalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n_model,
            y=y,
        )
    else:
        true_posterior = TruePosteriorGlobalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n_model,
            y=y,
        )

    ### generate guide
    if guide_type == "prior":
        guide = model
    elif guide_type == "qmc_prior":
        if model_type == "local_lvm":
            guide = QMCPriorLocalLVM(
                mu_z_init=mu_z_init,
                sigma_z_init=sigma_z_init,
            )
        else:
            guide = QMCPriorGlobalLVM(
                mu_z_init=mu_z_init,
                sigma_z_init=sigma_z_init,
            )
    elif guide_type == "true_posterior":
        guide = true_posterior
    elif guide_type == "approximate_posterior":
        guide = AutoNormal(model=model)
        # TODO: run guide once on the data to "show which sites are observed"
        # TODO: how to do this correctly, i.e., how to make sure guide does not
        # learn distribution over observed parameters?
        guide(y=y)

    ### record samples before training
    # sample z from prior
    all_samples_prior_init = predict(model=model, guide=None, L=L, N=N, S=S_plot)
    samples_obs_prior_init = all_samples_prior_init["obs"]
    samples_prior_init = all_samples_prior_init["z"]
    # sample z from true posterior
    all_samples_post_init = predict(
        model=model, guide=true_posterior, L=L, N=N, S=S_plot
    )
    samples_obs_post_init = all_samples_post_init["obs"]
    samples_post_init = all_samples_post_init["z"]
    # sample z from guide
    all_samples_guide_init = predict(model=model, guide=guide, L=L, N=N, S=S_plot)
    samples_obs_guide_init = all_samples_guide_init["obs"]
    samples_guide_init = all_samples_guide_init["z"]

    ### compute marginal likelihood before training
    N_S = 10
    marg_ll_sampled_init = torch.zeros(N_S)
    for i in range(N_S):
        marg_ll_sampled_init[i], _, _ = log_marginal_likelihood(
            model=model,
            guide=guide,
            y=y,
            S=S,
            estimator_type=lmlhd_estimator_type,
        )
    marg_ll_samp_mean_init = marg_ll_sampled_init.mean()
    marg_ll_samp_std_init = marg_ll_sampled_init.std()
    try:
        marg_ll_true_init = true_log_marginal_likelihood(model=model, y=y)
    except ValueError:
        marg_ll_true_init = None

    ### optimize prior
    if optimize:
        print(f"mu_z_opt  = {mu_z_opt:.4f}")
        print(f"std_z_opt = {sigma_z_opt:.4f}")
        losses = []
        true_elbos = []
        log_iws = []
        mu_zs = []
        sigma_zs = []
        grad_mu_z_unconstraineds = []
        grad_sigma_z_unconstraineds = []
        params = list(model.parameters())
        if guide_type == "approximate_posterior":
            params += list(guide.parameters())
        optim = torch.optim.Adam(lr=initial_lr, params=params)
        gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
        lr_decay = gamma ** (1 / n_epochs)
        lr_scheduler = ExponentialLR(optimizer=optim, gamma=lr_decay)
        for epoch in range(n_epochs):
            optim.zero_grad()
            lml, log_iw, true_elbo = log_marginal_likelihood(
                model=model,
                guide=guide,
                y=y,
                S=S,
                estimator_type=lmlhd_estimator_type,
            )
            loss = -lml
            loss.backward()

            # log
            losses.append(loss.item())
            true_elbos.append(true_elbo)
            mu_zs.append(model.mu_z.item())
            sigma_zs.append(model.sigma_z.item())
            grad_mu_z_unconstraineds.append(model.mu_z_unconstrained.grad.item())
            grad_sigma_z_unconstraineds.append(model.sigma_z_unconstrained.grad.item())
            log_iws.append(log_iw.detach())

            if epoch % 100 == 0 or epoch == n_epochs - 1:
                # print_pyro_parameters()
                string = f"epoch = {epoch:04d}" f" | loss = {loss.item():+.4f}"
                string += f" | mu_z = {model.mu_z.item():+.4f}"
                string += f" | mu_z = {model.sigma_z.item():+.4f}"
                # for name in pyro.get_param_store().keys():
                #     string += f" | {name} = {pyro.param(name)}"
                print(string)

            optim.step()
            lr_scheduler.step()

        ### record samples after training
        # sample z from prior
        all_samples_prior_trained = predict(model=model, guide=None, L=L, N=N, S=S_plot)
        samples_obs_prior_trained = all_samples_prior_trained["obs"]
        samples_prior_trained = all_samples_prior_trained["z"]
        # sample z from true posterior
        all_samples_post_trained = predict(
            model=model, guide=true_posterior, L=L, N=N, S=S_plot
        )
        samples_obs_post_trained = all_samples_post_trained["obs"]
        samples_post_trained = all_samples_post_trained["z"]
        # sample z from guide
        all_samples_guide_trained = predict(
            model=model, guide=guide, L=L, N=N, S=S_plot
        )
        samples_obs_guide_trained = all_samples_guide_trained["obs"]
        samples_guide_trained = all_samples_guide_trained["z"]

        ### compute marginal likelihood (trained)
        N_S = 10
        marg_ll_sampled_trained = torch.zeros(N_S)
        for i in range(N_S):
            marg_ll_sampled_trained[i], _, _ = log_marginal_likelihood(
                model=model,
                guide=guide,
                y=y,
                S=S,
                estimator_type=lmlhd_estimator_type,
            )
        marg_ll_samp_mean_trained = marg_ll_sampled_trained.mean()
        marg_ll_samp_std_trained = marg_ll_sampled_trained.std()
        try:
            marg_ll_true_trained = true_log_marginal_likelihood(model=model, y=y)
        except ValueError:
            marg_ll_true_trained = None

    ### print
    print("*" * 50)
    print(f"data mean                  = {y.mean():+.4f}")
    print(f"data std                   = {y.std():+.4f}")
    print(f"sigma_n (model)            = {model.sigma_n:+.4f}")
    print(f"sigma_n (true)             = {sigma_n_true:+.4f}")
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
    print(f"predictive mean (init)     = {samples_obs_prior_init.mean():+.4f}")
    print(f"predictive std (init)      = {samples_obs_prior_init.std():+.4f}")
    if optimize:
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
        print(f"predictive mean (trained)  = {samples_obs_prior_trained.mean():+.4f}")
        print(f"predictive std (trained)   = {samples_obs_prior_trained.std():+.4f}")
        print("*" * 20)
        print(f"prior mean (opt)           = {mu_z_opt:+.4f}")
        print(f"prior std (opt)            = {sigma_z_opt:+.4f}")
    print("*" * 50)

    ### plot prediction
    fig, axes = plt.subplots(nrows=1, ncols=5, squeeze=False, figsize=(15, 8))
    ax = axes[0, 0]
    for l in range(min(3, L)):
        plot_kde(
            y[l, :, 0],
            ax=ax,
            color="b",
            linestyle="-",
            label="data" if l == 0 else None,
        )
        for s in range(S_plot):
            plot_kde(
                samples_obs_prior_init[s, l, :, 0].reshape(-1),
                ax=ax,
                color="r",
                linestyle="--",
                alpha=0.3,
                label="obs prior (init)" if l == s == 0 else None,
            )
            plot_kde(
                samples_obs_post_init[s, l, :, 0].reshape(-1),
                ax=ax,
                color="g",
                linestyle="--",
                alpha=0.3,
                label="obs true posterior (init)" if l == s == 0 else None,
            )
            plot_kde(
                samples_obs_guide_init[s, l, :, 0].reshape(-1),
                ax=ax,
                color="y",
                linestyle="--",
                alpha=0.3,
                label="obs guide (init)" if l == s == 0 else None,
            )
    ax.set_title("Data and Predictions (init)")
    ax.set_xlabel("y")
    ax.set_ylabel("p(y)")
    ax.grid()
    ax.legend()

    ax = axes[0, 1]
    for l in range(L_plot):
        plot_functions(
            y[l, :, 0],
            ax=ax,
            color="b",
            linestyle="-",
            label="data" if l == 0 else None,
        )
        for s in range(S_plot):
            plot_functions(
                samples_obs_prior_init[s, l, :, 0].reshape(-1),
                ax=ax,
                color="r",
                linestyle="--",
                alpha=0.3,
                label="obs prior (init)" if l == s == 0 else None,
            )
            plot_functions(
                samples_obs_post_init[s, l, :, 0].reshape(-1),
                ax=ax,
                color="g",
                linestyle="--",
                alpha=0.3,
                label="obs true posterior (init)" if l == s == 0 else None,
            )
            plot_functions(
                samples_obs_guide_init[s, l, :, 0].reshape(-1),
                ax=ax,
                color="y",
                linestyle="--",
                alpha=0.3,
                label="obs guide (init)" if l == s == 0 else None,
            )
    ax.set_title("Data and Predictions (init)")
    ax.set_xlabel("n")
    ax.set_ylabel("y_n")
    ax.grid()
    ax.legend()

    if optimize:
        ax = axes[0, 1]
        ax.sharex(axes[0, 0])
        for l in range(min(3, L)):
            plot_kde(
                y[l, :, 0],
                ax=ax,
                color="b",
                linestyle="-",
                label="data" if l == 0 else None,
            )
            for s in range(S_plot):
                plot_kde(
                    samples_obs_prior_trained[s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="r",
                    linestyle="-",
                    alpha=0.3,
                    label="obs prior (trained)" if l == s == 0 else None,
                )
                plot_kde(
                    samples_obs_post_trained[s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="g",
                    linestyle="-",
                    alpha=0.3,
                    label="obs true posterior (trained)" if l == s == 0 else None,
                )
                plot_kde(
                    samples_obs_guide_trained[s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="y",
                    linestyle="-",
                    alpha=0.3,
                    label="obs guide (trained)" if l == s == 0 else None,
                )
        ax.set_title("Data and Predictions (trained)")
        ax.set_xlabel("y_i")
        ax.set_ylabel("p(y_i)")
        ax.grid()
        ax.legend()

        ax = axes[0, 2]
        ax.plot(np.arange(len(losses)), losses, label="loss")
        if true_elbos[0] is not None:
            ax.plot(np.arange(len(true_elbos)), true_elbos, label="true elbo")
        ax.set_title("Learning curve")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid()
        ax.legend()

        ax = axes[0, 3]
        ax.scatter(x=mu_zs, y=sigma_zs, c=np.arange(n_epochs) + 1)
        ax.set_title("Learning trajectory")
        ax.axvline(mu_z_opt, ls="--")
        ax.axhline(sigma_z_opt, ls="--")
        ax.set_xlabel("mu_z")
        ax.set_ylabel("sigma_z")
        ax.grid()
        plt.tight_layout()

        log_iw = torch.stack(log_iws, dim=0)
        log_iw = log_iw.reshape(n_epochs, -1)
        fig, axes = plt.subplots(
            nrows=5, ncols=5, figsize=(15, 8), squeeze=False, sharex=True, sharey=True
        )
        fig.suptitle("Distribution of Log Importance Weights")
        for i, epoch in enumerate(range(0, n_epochs, n_epochs // 25)):
            ax = axes[i // 5, i % 5]
            sns.histplot(data=log_iw[epoch], ax=ax, bins=25)
            ax.set_title(f"ep = {epoch:d}")
        plt.tight_layout()

        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(15, 8), squeeze=False, sharex=True, sharey=True
        )
        fig.suptitle("Gradients of mu_z == mu_z_unconstrained")
        ax = axes[0, 0]
        ax.plot(np.arange(len(grad_mu_z_unconstraineds)), grad_mu_z_unconstraineds)
        ax.set_ylabel(f"grad_mu_z")
        ax.set_xlabel(f"epoch")
        ax.grid()

        fig.suptitle("Gradients of sigma_z_unconstrained")
        ax = axes[1, 0]
        ax.plot(
            np.arange(len(grad_sigma_z_unconstraineds)), grad_sigma_z_unconstraineds
        )
        ax.set_ylabel(f"grad_sigma_z_unconstrained")
        ax.set_xlabel(f"epoch")
        ax.grid()

        plt.tight_layout()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
