import math

import numpy as np
import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.autograd import grad
from mtutils.mtutils import print_pyro_parameters
from numpy import dtype, mod
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
allowed_log_marginal_likelihood_estimator_types = ["standard_elbo", "iwae_elbo"]


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


def sampled_log_marginal_likelihood(model, guide, y, S, estimator_type):
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

    assert estimator_type in allowed_log_marginal_likelihood_estimator_types
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


def get_optimal_prior_parameters(model_type, sigma_n, y):
    # TODO: solution for global LVM is not accurate
    from sklearn.covariance import EmpiricalCovariance

    L = y.shape[0]
    N = y.shape[1]

    mu_emp = y.mean(dim=0)
    Sigma_emp = EmpiricalCovariance().fit(y.squeeze(-1))

    mu_z_opt = mu_emp.mean()
    if model_type == "local_lvm":
        var_opt = 1 / N * np.trace(Sigma_emp.covariance_) - sigma_n ** 2
        sigma_z_opt = np.sqrt(var_opt)
    else:
        assert model_type == "global_lvm"
        sigma_z_opt = np.sqrt(Sigma_emp.covariance_[0, 1])  # TODO: not accurate for N>2

    return mu_z_opt, sigma_z_opt


def generate_model(model_type, mu_z_init, sigma_z_init, sigma_n):
    if model_type == "local_lvm":
        model = LocalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n,
        )
    elif model_type == "global_lvm":
        model = GlobalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n,
        )

    return model


def generate_true_posterior(model, mu_z_init, sigma_z_init, sigma_n, y):
    if isinstance(model, LocalLVM):
        true_posterior = TruePosteriorLocalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n,
            y=y,
        )
    elif isinstance(model, GlobalLVM):
        true_posterior = TruePosteriorGlobalLVM(
            mu_z_init=mu_z_init,
            sigma_z_init=sigma_z_init,
            sigma_n=sigma_n,
            y=y,
        )

    return true_posterior


def generate_guide(model, guide_type, mu_z_init, sigma_z_init, sigma_n=None, y=None):
    if guide_type == "prior":
        if isinstance(model, LocalLVM):
            guide = LocalLVMPrior(mu_z_init=mu_z_init, sigma_z_init=sigma_z_init)
        elif isinstance(model, GlobalLVM):
            guide = GlobalLVMPrior(mu_z_init=mu_z_init, sigma_z_init=sigma_z_init)
    elif guide_type == "qmc_prior":
        if isinstance(model, LocalLVM):
            guide = LocalLVMQMCPrior(mu_z_init=mu_z_init, sigma_z_init=sigma_z_init)
        elif isinstance(model, GlobalLVM):
            guide = GlobalLVMQMCPrior(mu_z_init=mu_z_init, sigma_z_init=sigma_z_init)
    elif guide_type == "true_posterior":
        assert sigma_n is not None
        assert y is not None
        if isinstance(model, LocalLVM):
            guide = LocalLVMTruePosterior(
                mu_z_init=mu_z_init, sigma_z_init=sigma_z_init, sigma_n=sigma_n, y=y
            )
        elif isinstance(model, GlobalLVM):
            guide = GlobalLVMTruePosterior(
                mu_z_init=mu_z_init, sigma_z_init=sigma_z_init, sigma_n=sigma_n, y=y
            )
    elif guide_type == "approximate_posterior":
        guide = AutoNormal(model=model)
        # Run guide once on the data to "show which sites are observed"
        # TODO: how to do this correctly, i.e., how to make sure guide does not
        # learn distribution over observed parameters?
        guide(y=y)

    return guide


def get_samples(model, guide, true_posterior, L, N, S):
    samples_prior = predict(model=model, guide=None, L=L, N=N, S=S)
    samples_true_posterior = predict(model=model, guide=true_posterior, L=L, N=N, S=S)
    samples_guide = predict(model=model, guide=guide, L=L, N=N, S=S)

    return samples_prior, samples_true_posterior, samples_guide


def get_log_marginal_likelihood(model, guide, y, S, N_S, estimator_type):
    ## sampled solution
    lml_sampled = torch.zeros(N_S)
    for i in range(N_S):
        lml_sampled[i], _, _ = sampled_log_marginal_likelihood(
            model=model,
            guide=guide,
            y=y,
            S=S,
            estimator_type=estimator_type,
        )
    lml_sampled_mean = lml_sampled.mean()
    lml_sampled_std = lml_sampled.std()

    ## analytical solution
    try:
        lml_true = true_log_marginal_likelihood(model=model, y=y)
    except ValueError:
        lml_true = None

    return lml_sampled_mean, lml_sampled_std, lml_true


def optimize_prior(
    model, guide, y, S, lml_estimator_type, n_epochs, initial_lr, final_lr
):
    # prepare return values
    losses = []
    pyro_elbos = []
    log_iws = []
    mu_zs = []
    sigma_zs = []

    # prepare optimizer
    params = set(model.parameters() + guide.parameters())
    optim = torch.optim.Adam(lr=initial_lr, params=params)
    gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
    lr_decay = gamma ** (1 / n_epochs)
    lr_scheduler = ExponentialLR(optimizer=optim, gamma=lr_decay)

    # optimization loop
    for epoch in range(n_epochs):
        optim.zero_grad()
        lml, log_iw, pyro_elbo = sampled_log_marginal_likelihood(
            model=model,
            guide=guide,
            y=y,
            S=S,
            estimator_type=lml_estimator_type,
        )
        loss = -lml
        loss.backward()

        # log
        losses.append(loss.item())
        pyro_elbos.append(pyro_elbo)
        mu_zs.append(model.mu_z.item())
        sigma_zs.append(model.sigma_z.item())
        log_iws.append(log_iw.detach())
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            string = f"epoch = {epoch:04d}" f" | loss = {loss.item():+.4f}"
            string += f" | mu_z = {model.mu_z.item():+.4f}"
            string += f" | mu_z = {model.sigma_z.item():+.4f}"
            print(string)

        optim.step()
        lr_scheduler.step()

    return losses, pyro_elbos, mu_zs, sigma_zs, log_iws


def log_results(
    model,
    y,
    mu_z_true,
    sigma_z_true,
    sigma_n_true,
    mu_z_init,
    sigma_z_init,
    mu_z_opt,
    sigma_z_opt,
    lml_true_init,
    lml_sampled_init_mean,
    lml_sampled_init_std,
    lml_true_trained=None,
    lml_sampled_trained_mean=None,
    lml_sampled_trained_std=None,
):
    ### print
    print("*" * 50)

    # data
    print(f"*" * 20)
    print(f"mu_z_true                  = {mu_z_true:+.4f}")
    print(f"sigma_z_true               = {sigma_z_true:+.4f}")
    print(f"sigma_n_true               = {sigma_n_true:+.4f}")
    print(f"data mean                  = {y.mean():+.4f}")
    print(f"data std                   = {y.std():+.4f}")

    print("*" * 20)
    print(f"mu_z_opt                   = {mu_z_opt:+.4f}")
    print(f"sigma_z_opt                = {sigma_z_opt:+.4f}")

    print("*" * 20)
    print(f"mu_z_init                  = {mu_z_init:+.4f}")
    print(f"sigma_z_init               = {sigma_z_init:+.4f}")
    print(f"sigma_n_model              = {model.sigma_n:+.4f}")
    print(
        f"lml (init, sampled)       = {lml_sampled_init_mean:+.4f} "
        f"+- {lml_sampled_init_std:.4f}"
    )
    if lml_true_init is not None:
        print(f"lml (init, true)           = {lml_true_init:+.4f}")
    else:
        print(f"lml (init, true)           = computation error")

    if lml_sampled_trained_mean is not None:  # if model was trained
        print("*" * 20)
        print(f"mu_z_trained               = {model.mu_z.item():+.4f}")
        print(f"sigma_z_trained            = {model.sigma_z.item():+.4f}")
        print(
            f"lml     (trained, sampled) = {lml_sampled_trained_mean:+.4f} "
            f"+- {lml_sampled_trained_std:.4f}"
        )
        if lml_true_trained is not None:
            print(f"lml (trained, true)       = {lml_true_trained:+.4f}")
        else:
            print(f"lml (trained, true)       = computation error")
    else:
        assert lml_sampled_trained_std is None
        assert lml_true_trained is None

    print("*" * 50)


def plot_summary(
    y,
    samples_prior_init,
    samples_true_posterior_init,
    samples_guide_init,
    samples_prior_trained=None,
    samples_true_posterior_trained=None,
    samples_guide_trained=None,
    losses=None,
    pyro_elbos=None,  # elbos computed by Pyro's implementation
    mu_zs=None,
    sigma_zs=None,
    mu_z_opt=None,
    sigma_z_opt=None,
):
    L = y.shape[0]
    S = samples_prior_init["obs"].shape[0]

    ## prepare plot
    optimization_performed = samples_prior_trained is not None
    nrows = 2 if optimization_performed else 1
    ncols = 3 if optimization_performed else 2
    fig, axes = plt.subplots(nrows=2, ncols=3, squeeze=False, figsize=(15, 8))

    ## initial data and predictions (distributions)
    ax = axes[0, 0]
    for l in range(min(3, L)):
        plot_kde(
            y[l, :, 0],
            ax=ax,
            color="b",
            linestyle="-",
            label="data" if l == 0 else None,
        )
        for s in range(S):
            plot_kde(
                samples_prior_init["obs"][s, l, :, 0].reshape(-1),
                ax=ax,
                color="r",
                linestyle="-",
                alpha=0.3,
                label="obs prior (init)" if l == s == 0 else None,
            )
            plot_kde(
                samples_true_posterior_init["obs"][s, l, :, 0].reshape(-1),
                ax=ax,
                color="g",
                linestyle="-",
                alpha=0.3,
                label="obs true posterior (init)" if l == s == 0 else None,
            )
            plot_kde(
                samples_guide_init["obs"][s, l, :, 0].reshape(-1),
                ax=ax,
                color="y",
                linestyle="-",
                alpha=0.3,
                label="obs guide (init)" if l == s == 0 else None,
            )
    ax.set_title("Data and Predictions (init)")
    ax.set_xlabel("y")
    ax.set_ylabel("p(y)")
    ax.grid()
    ax.legend()

    ## initial data and predictions (functions)
    ax = axes[0, 1]
    for l in range(L):
        plot_functions(
            y[l, :, 0],
            ax=ax,
            color="b",
            linestyle="-",
            label="data" if l == 0 else None,
        )
        for s in range(S):
            plot_functions(
                samples_prior_init["obs"][s, l, :, 0].reshape(-1),
                ax=ax,
                color="r",
                linestyle="-",
                alpha=0.3,
                label="obs prior (init)" if l == s == 0 else None,
            )
            plot_functions(
                samples_true_posterior_init["obs"][s, l, :, 0].reshape(-1),
                ax=ax,
                color="g",
                linestyle="-",
                alpha=0.3,
                label="obs true posterior (init)" if l == s == 0 else None,
            )
            plot_functions(
                samples_guide_init["obs"][s, l, :, 0].reshape(-1),
                ax=ax,
                color="y",
                linestyle="-",
                alpha=0.3,
                label="obs guide (init)" if l == s == 0 else None,
            )
    ax.set_title("Data and Predictions (init)")
    ax.set_xlabel("n")
    ax.set_ylabel("y_n")
    ax.grid()
    ax.legend()

    if optimization_performed:  # only if model was trained
        ## trained data and predictions (distributions)
        ax = axes[1, 0]
        ax.sharex(axes[0, 0])
        for l in range(L):
            plot_kde(
                y[l, :, 0],
                ax=ax,
                color="b",
                linestyle="-",
                label="data" if l == 0 else None,
            )
            for s in range(S):
                plot_kde(
                    samples_prior_trained["obs"][s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="r",
                    linestyle="-",
                    alpha=0.3,
                    label="obs prior (trained)" if l == s == 0 else None,
                )
                plot_kde(
                    samples_true_posterior_trained["obs"][s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="g",
                    linestyle="-",
                    alpha=0.3,
                    label="obs true posterior (trained)" if l == s == 0 else None,
                )
                plot_kde(
                    samples_guide_trained["obs"][s, l, :, 0].reshape(-1),
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

        ## trained data and predictions (functions)
        ax = axes[1, 1]
        ax.sharex(axes[0, 1])
        ax.sharey(axes[0, 1])
        for l in range(L):
            plot_functions(
                y[l, :, 0],
                ax=ax,
                color="b",
                linestyle="-",
                label="data" if l == 0 else None,
            )
            for s in range(S):
                plot_functions(
                    samples_prior_trained["obs"][s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="r",
                    linestyle="-",
                    alpha=0.3,
                    label="obs prior (trained)" if l == s == 0 else None,
                )
                plot_functions(
                    samples_true_posterior_trained["obs"][s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="g",
                    linestyle="-",
                    alpha=0.3,
                    label="obs true posterior (trained)" if l == s == 0 else None,
                )
                plot_functions(
                    samples_guide_trained["obs"][s, l, :, 0].reshape(-1),
                    ax=ax,
                    color="y",
                    linestyle="-",
                    alpha=0.3,
                    label="obs guide (trained)" if l == s == 0 else None,
                )
        ax.set_title("Data and Predictions (trained)")
        ax.set_xlabel("n")
        ax.set_ylabel("y_n")
        ax.grid()
        ax.legend()

        ## Learning curve
        ax = axes[0, 2]
        ax.plot(np.arange(len(losses)), losses, label="loss")
        if pyro_elbos[0] is not None:
            ax.plot(np.arange(len(pyro_elbos)), pyro_elbos, label="Pyro elbo")
        ax.set_title("Learning curve")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid()
        ax.legend()

        ## Learning trajectory
        ax = axes[1, 2]
        n_epochs = len(mu_zs)
        ax.scatter(x=mu_zs, y=sigma_zs, c=np.arange(n_epochs) + 1)
        ax.set_title("Learning trajectory")
        ax.axvline(mu_z_opt, ls="--")
        ax.axhline(sigma_z_opt, ls="--")
        ax.set_xlabel("mu_z")
        ax.set_ylabel("sigma_z")
        ax.grid()
        plt.tight_layout()

    plt.tight_layout()


def plot_log_importance_weight_histograms(log_iws):
    n_epochs = len(log_iws)
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


def run_experiment(config, wandb_run):
    ### check some settings
    assert config["model_type"] in allowed_model_types
    assert config["guide_type"] in allowed_guide_types
    assert (
        config["log_marginal_likelihood_estimator_type"]
        in allowed_log_marginal_likelihood_estimator_types
    )
    # the optimal solutions depend on this:
    assert config["sigma_n_model"] == config["sigma_n_true"]

    ### seed torch, random, np.random
    pyro.set_rng_seed(config["rng_seed"])

    ### data
    y = generate_data(
        L=config["L"],
        N=config["N"],
        sigma_n=config["sigma_n_true"],
        mu_z=config["mu_z_true"],
        sigma_z=config["sigma_z_true"],
    )

    ### optimal prior parameters
    mu_z_opt, sigma_z_opt = get_optimal_prior_parameters(
        model_type=config["model_type"],
        sigma_n=config["sigma_n_true"],
        y=y,
    )

    ### model
    model = generate_model(
        model_type=config["model_type"],
        mu_z_init=config["mu_z_init"],
        sigma_z_init=config["sigma_z_init"],
        sigma_n=config["sigma_n_model"],
    )

    ### true posterior
    true_posterior = generate_true_posterior(
        model_type=config["model_type"],
        mu_z_init=config["mu_z_init"],
        sigma_z_init=config["sigma_z_init"],
        sigma_n=config["sigma_n_model"],
        y=y,
    )

    ### guide
    guide = generate_guide(
        model=model,
        guide_type=config["guide_type"],
        mu_z_init=config["mu_z_init"],
        sigma_z_init=config["sigma_z_init"],
        # only required if guide_type == "true_posterior":
        sigma_n=config["sigma_n_model"],
        y=y,
    )

    ### record samples before training
    samples_prior_init, samples_true_posterior_init, samples_guide_init = get_samples(
        model=model,
        guide=guide,
        L=config["L_plot"],
        S=config["S_plot"],
    )

    ### compute log marginal likelihood before training
    (
        lml_sampled_init_mean,
        lml_sampled_init_std,
        lml_true_init,
    ) = get_log_marginal_likelihood(
        model=model,
        guide=guide,
        y=y,
        S=config["S"],
        N_S=config["N_S"],
        estimator_type=config["lml_estimator_type"],
    )

    ### optimize prior
    if config["optimize"]:
        print(f"mu_z_opt  = {mu_z_opt:.4f}")
        print(f"std_z_opt = {sigma_z_opt:.4f}")
        losses, pyro_elbos, mu_zs, sigma_zs, log_iws = optimize_prior(
            model=model,
            guide=guide,
            y=y,
            S=config["S"],
            lml_estimator_type=config["lml_estimator_type"],
            n_epochs=config["n_epochs"],
            initial_lr=config["initial_lr"],
            final_lr=config["final_lr"],
        )

        ### record samples after training
        (
            samples_prior_trained,
            samples_true_posterior_trained,
            samples_guide_trained,
        ) = get_samples(
            model=model,
            guide=guide,
            L=config["L_plot"],
            S=config["S_plot"],
        )

        ### compute marginal likelihood after training
        (
            lml_sampled_trained_mean,
            lml_sampled_trained_std,
            lml_true_trained,
        ) = get_log_marginal_likelihood(
            model=model,
            guide=guide,
            y=y,
            S=config["S"],
            N_S=config["N_S"],
            estimator_type=config["lml_estimator_type"],
        )

    ## log results
    log_results(
        model=model,
        y=y,
        mu_z_true=config["mu_z_true"],
        sigma_z_true=config["sigma_z_true"],
        mu_z_init=config["mu_z_init"],
        sigma_z_init=config["sigma_z_init"],
        mu_z_opt=config["mu_z_opt"],
        sigma_z_opt=config["sigma_z_opt"],
        lml_true_init=lml_true_init,
        lml_sampled_init_mean=lml_sampled_init_mean,
        lml_sampled_init_std=lml_sampled_init_std,
        lml_true_trained=lml_true_trained if config["optimize"] else None,
        lml_sampled_trained_mean=lml_sampled_trained_mean
        if config["optimize"]
        else None,
        lml_sampled_trained_std=lml_sampled_trained_std if config["optimize"] else None,
    )

    ## plot results
    plot_summary(
        y=y,
        samples_prior_init=samples_prior_init,
        samples_true_posterior_init=samples_true_posterior_init,
        samples_guide_init=samples_guide_init,
        samples_prior_trained=samples_prior_trained if config["optimize"] else None,
        samples_true_posterior_trained=samples_true_posterior_trained
        if config["optimize"]
        else None,
        samples_guide_trained=samples_guide_trained if config["optimize"] else None,
        losses=losses if config["optimize"] else None,
        pyro_elbos=pyro_elbos if config["optimize"] else None,
        mu_zs=mu_zs if config["optimize"] else None,
        sigma_zs=sigma_zs if config["sigma_zs"] else None,
        mu_z_opt=mu_z_opt if config["optimize"] else None,
        sigma_z_opt=sigma_z_opt if config["optimize"] else None,
    )
    if config["optimize"]:
        plot_log_importance_weight_histograms(log_iws=log_iws)
    plt.show()


def main():
    ### settings
    pyro.set_rng_seed(123)
    optimize = True
    n_epochs = 1000
    initial_lr = 1e-2
    final_lr = 1e-2
    S_plot = 10
    S = 2 ** 0  # number of samples for log marg likelihood / gradient estimation
    N_S = 25  # number of sample sets for log marg likelihood estimation
    ## data
    L = 10
    N = 50
    mu_z_true = 1.0
    sigma_z_true = 0.5
    sigma_n_true = 0.01
    ## model
    # model_type = "local_lvm"
    model_type = "global_lvm"
    mu_z_init = -1.0
    sigma_z_init = 1.0
    # mu_z_init = mu_z_true
    # sigma_z_init = sigma_z_true
    sigma_n_model = sigma_n_true
    ## guide
    guide_type = "prior"
    # guide_type = "qmc_prior"  # understand scrambling
    # guide_type = "true_posterior"
    # guide_type = "approximate_posterior"
    ## log marginal likelihood estimator
    lmlhd_estimator_type = "iwae_elbo"
    # lmlhd_estimator_type = "standard_elbo"


if __name__ == "__main__":
    main()
