import copy
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from metalearning_benchmarks import Affine1D, Linear1D, Quadratic1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from numpy.core.fromnumeric import sort
from pyro import distributions as dist
from pyro import poutine
from pyro.distributions import constraints
from pyro.infer import SVI, Predictive, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.optim import ClippedAdam
from torch import nn
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam as AdamTorch
from torch.optim.lr_scheduler import ExponentialLR

### PyroNotes:
## Learned PyroParams
# https://docs.pyro.ai/en/dev/nn.html#pyro.nn.module.PyroSample
# https://forum.pyro.ai/t/getting-estimates-of-parameters-that-use-pyrosample/2901/2
## Plates with explicit independent dimensions
# https://pyro.ai/examples/tensor_shapes.html#Declaring-independent-dims-with-plate


class MTBayesianLinear(PyroModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_type: str = "isotropic",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_type = prior_type

        ## weight prior
        if self.prior_type == "isotropic":
            self.weight_prior_loc = PyroParam(
                init_value=torch.tensor(0.0),
                constraint=constraints.real,
            )
            self.weight_prior_scale = PyroParam(
                init_value=torch.tensor(1.0),
                constraint=constraints.positive,
            )
            self.weight_prior = (
                lambda self: dist.Normal(self.weight_prior_loc, self.weight_prior_scale)
                .expand([self.out_features, self.in_features])
                .to_event(2)
            )
        elif self.prior_type == "diagonal":
            self.weight_prior_loc = PyroParam(
                init_value=torch.zeros(self.out_features, self.in_features),
                constraint=constraints.real,
            )
            self.weight_prior_scale = PyroParam(
                init_value=torch.ones(self.out_features, self.in_features),
                constraint=constraints.positive,
            )
            self.weight_prior = lambda self: dist.Normal(
                self.weight_prior_loc, self.weight_prior_scale
            ).to_event(2)
        else:
            raise ValueError(f"Unknown prior specification '{self.prior_type}'!")
        self.weight = PyroSample(self.weight_prior)

        ## bias prior
        if bias:
            if self.prior_type == "isotropic":
                self.bias_prior_loc = PyroParam(
                    init_value=torch.tensor(0.0),
                    constraint=constraints.real,
                )
                self.bias_prior_scale = PyroParam(
                    init_value=torch.tensor(1.0),
                    constraint=constraints.positive,
                )
                self.bias_prior = (
                    lambda self: dist.Normal(self.bias_prior_loc, self.bias_prior_scale)
                    .expand([self.out_features])
                    .to_event(1)
                )
            elif self.prior_type == "diagonal":
                self.bias_prior_loc = PyroParam(
                    init_value=torch.zeros(self.out_features),
                    constraint=constraints.real,
                )
                self.bias_prior_scale = PyroParam(
                    init_value=torch.ones(self.out_features),
                    constraint=constraints.positive,
                )
                self.bias_prior = lambda self: dist.Normal(
                    self.bias_prior_loc, self.bias_prior_scale
                ).to_event(1)
            else:
                raise ValueError(f"Unknown prior specification '{self.prior_type}'!")
            self.bias = PyroSample(self.bias_prior)
        else:
            self.bias = None

    def forward(self, x):
        x = x.detach().clone()
        weight = self.weight.detach().clone()
        bias = self.bias.detach().clone()

        ## check shapes
        # x.shape = (n_tasks, n_points, d_x)
        assert x.ndim == 3
        n_tasks = x.shape[0]

        # weight.event_shape == (self.out_features, self.in_features)
        # weight.batch_shape depends on whether a sample dimension is added, (e.g.,
        #  by Predictive)
        has_sample_dim = len(weight.shape) == 5
        if not has_sample_dim:
            n_samples = 1
            assert weight.shape == (
                n_tasks,
                1,  # because the n_pts plate is nested inside of the n_tsk plate
                self.out_features,
                self.in_features,
            )
            # add sample dim
            weight = weight[None, :, :, :, :]
        else:
            n_samples = weight.shape[0]
        # squeeze n_pts batch dimension
        weight = weight.squeeze(2)
        assert weight.shape == (
            n_samples,
            n_tasks,
            self.out_features,
            self.in_features,
        )

        # expand x to sample shape
        x = x.expand(torch.Size([n_samples]) + x.shape)

        ## compute the linear transformation
        # TODO: why do we need x.float() here as soon as we have more than zero layers?
        y = torch.einsum("slyx,slnx->slny", weight, x.float())

        if bias is not None:
            ## check shapes
            # bias.event_shape = (self.out_features)
            # bias.batch_shape = (n_tasks, 1) or (n_samples, n_tasks, 1) (cf. above)
            if has_sample_dim:
                assert bias.shape == (n_samples, n_tasks, 1, self.out_features)
            else:
                assert bias.shape == (n_tasks, 1, self.out_features)
                # add sample dim
                bias = bias[None, :, :, :]
            # squeeze the n_pts batch dimension
            bias = bias.squeeze(2)
            assert bias.shape == (n_samples, n_tasks, self.out_features)

            ## add the bias
            y = y + bias[:, :, None, :]

        if not has_sample_dim:
            y.squeeze_(
                0
            )  # if we do not have a sample dimension, we must not return one

        return y

    def get_prior_distribution(self):
        return [self.weight_prior(self), self.bias_prior(self)]


class MTBNN(PyroModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden: int,
        d_hidden: int,
        prior_type: str,
    ):
        super().__init__()

        self.n_hidden = n_hidden
        modules = []
        if self.n_hidden == 0:
            modules.append(
                MTBayesianLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=True,
                    prior_type=prior_type,
                )
            )
        else:
            modules.append(
                MTBayesianLinear(
                    in_features=in_features,
                    out_features=d_hidden,
                    bias=True,
                    prior_type=prior_type,
                )
            )
            modules.append(PyroModule[nn.Tanh]())
            for _ in range(self.n_hidden - 1):
                modules.append(
                    MTBayesianLinear(
                        in_features=d_hidden,
                        out_features=d_hidden,
                        bias=True,
                        prior_type=prior_type,
                    )
                )
                modules.append(PyroModule[nn.Tanh]())
            modules.append(
                MTBayesianLinear(
                    in_features=d_hidden,
                    out_features=out_features,
                    bias=True,
                    prior_type=prior_type,
                )
            )
        self.net = PyroModule[nn.Sequential](*modules)

    def forward(self, x):
        return self.net(x)

    def freeze_prior(self):
        # freeze the unconstrained parameters
        # -> those are the leaf variables of the autograd graph
        # -> those are the registered parameters of self
        for p in self.parameters():
            p.requires_grad = False

    def get_prior_distribution(self):
        prior_distribution = []
        for module in self.net:
            if hasattr(module, "get_prior_distribution"):
                prior_distribution += module.get_prior_distribution()
        return prior_distribution


class MTBayesianRegression(PyroModule):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        n_hidden: int,
        d_hidden: int,
        noise_stddev: Optional[float] = None,
        prior_type: str = "isotropic",
    ):
        super().__init__()

        ## multi-task BNN
        self.mtbnn = MTBNN(
            in_features=d_x,
            out_features=d_y,
            n_hidden=n_hidden,
            d_hidden=d_hidden,
            prior_type=prior_type,
        )

        ## noise stddev
        # TODO: learn noise prior?
        self.noise_stddev_prior = (
            dist.Uniform(0.0, 1.0) if noise_stddev is None else None
        )
        self.noise_stddev = (
            PyroSample(self.noise_stddev_prior)
            if noise_stddev is None
            else noise_stddev
        )

    def forward(self, x: torch.tensor, y: Optional[torch.tensor] = None):
        # shapes
        assert x.ndim == 3
        if y is not None:
            assert y.ndim == 3
        n_tasks = x.shape[0]
        n_points = x.shape[1]

        noise_stddev = self.noise_stddev  # (sample) noise stddev
        with pyro.plate("tasks", n_tasks, dim=-2):
            mean = self.mtbnn(x)  # sample weights and compute mean pred
            # noise stddev can have a sample dimension! -> expand to mean's shape
            noise_stddev = noise_stddev.reshape([-1] + [1] * (mean.ndim - 1))
            noise_stddev = noise_stddev.expand(mean.shape)
            with pyro.plate("data", n_points, dim=-1):
                obs = pyro.sample(
                    "obs", dist.Normal(mean, noise_stddev).to_event(1), obs=y
                )  # score mean predictions against ground truth
        return mean

    def freeze_prior(self):
        self.mtbnn.freeze_prior()

    def get_prior_distribution(self):
        return self.mtbnn.get_prior_distribution()


def predict(model, guide, x: np.ndarray, n_samples: int):
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        parallel=True,  # our model is vectorized
        return_sites=(
            # if model is linear, those sites are available
            "mtbnn.net.0.weight",
            "mtbnn.net.0.bias",
            # those sites are always available
            "obs",
            "sigma",
            "_RETURN",
        ),
    )
    svi_samples = predictive(x=torch.tensor(x), y=None)
    svi_samples = {k: v for k, v in svi_samples.items()}
    pred_summary = summary(svi_samples)

    return pred_summary, svi_samples


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0).detach().cpu().numpy(),
            "std": torch.std(v, 0).detach().cpu().numpy(),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0].detach().cpu().numpy(),
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0].detach().cpu().numpy(),
        }
    return site_stats


def plot_predictions_for_one_set_of_tasks(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    pred_summary: dict,
    plot_obs: bool,
    ax: None,
    max_tasks: int,
    x_context: Optional[np.ndarray] = None,
    y_context: Optional[np.ndarray] = None,
):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

    x_pred = x_pred.squeeze(-1)
    means_plt = pred_summary["_RETURN"]["mean"].squeeze(-1)
    means_perc5_plt = pred_summary["_RETURN"]["5%"].squeeze(-1)
    means_perc95_plt = pred_summary["_RETURN"]["95%"].squeeze(-1)
    obs_plt = pred_summary["obs"]["mean"].squeeze(-1)
    obs_perc5_plt = pred_summary["obs"]["5%"].squeeze(-1)
    obs_perc95_plt = pred_summary["obs"]["95%"].squeeze(-1)

    n_task = x.shape[0]
    for l in range(n_task):
        if l == max_tasks:
            break
        line = ax.plot(x_pred[l, :], means_plt[l, :])[0]
        ax.scatter(x[l, :], y[l, :], color=line.get_color())
        if x_context is not None:
            ax.scatter(
                x_context[l, :],
                y_context[l, :],
                marker="x",
                s=100,
                color=line.get_color(),
            )
        if not plot_obs:
            ax.fill_between(
                x_pred[l, :],
                means_perc5_plt[l, :],
                means_perc95_plt[l, :],
                alpha=0.3,
                color=line.get_color(),
            )
        else:
            ax.fill_between(
                x_pred[l, :],
                obs_perc5_plt[l, :],
                obs_perc95_plt[l, :],
                alpha=0.3,
                color=line.get_color(),
            )
    ax.grid()


def plot_predictions(
    x_meta,
    y_meta,
    x_pred_meta,
    x_test_context,
    y_test_context,
    x_test_target,
    y_test_target,
    x_pred_test,
    pred_summary_prior_meta_untrained,
    pred_summary_prior_meta_trained,
    pred_summary_posterior_meta,
    pred_summary_posterior_test,
    max_tasks=3,
):
    assert x_meta.shape[-1] == y_meta.shape[-1] == x_pred_meta.shape[-1]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharey=True)
    fig.suptitle(f"Prior and Posterior Predictions")

    ax = axes[0, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean + Meta Data")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_untrained,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean\n(trained on meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_trained,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Mean\n(meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_posterior_meta,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 3]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Mean\n(test data)")
    plot_predictions_for_one_set_of_tasks(
        x=np.concatenate((x_test_context, x_test_target), axis=1),
        y=np.concatenate((y_test_context, y_test_target), axis=1),
        x_pred=x_pred_test,
        x_context=x_test_context,
        y_context=y_test_context,
        pred_summary=pred_summary_posterior_test,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[1, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation + Meta Data")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_untrained,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation\n(trained on meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_prior_meta_trained,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Observation\n(meta data)")
    plot_predictions_for_one_set_of_tasks(
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        pred_summary=pred_summary_posterior_meta,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 3]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Observation\n(test data)")
    plot_predictions_for_one_set_of_tasks(
        x=np.concatenate((x_test_context, x_test_target), axis=1),
        y=np.concatenate((y_test_context, y_test_target), axis=1),
        x_pred=x_pred_test,
        x_context=x_test_context,
        y_context=y_test_context,
        pred_summary=pred_summary_posterior_test,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )
    fig.tight_layout()


def plot_distributions(
    site_name,
    bm_meta,
    bm_test,
    bm_param_idx,
    samples_prior_meta_untrained,
    samples_prior_meta_trained,
    samples_posterior_meta,
    samples_posterior_test,
    max_tasks=3,
):
    # TODO: why do we need to squeeze the slope samples?
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Prior and Posterior Distributions of site '{site_name}'")

    for l, task in enumerate(bm_meta):
        if l == max_tasks:
            break
        ax = axes[0]
        ax.set_title("Prior distribution\n(untrained)")
        sns.distplot(
            samples_prior_meta_untrained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])

        ax = axes[1]
        ax.set_title("Prior distribution\n(trained on meta data)")
        sns.distplot(
            samples_prior_meta_trained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])

        ax = axes[2]
        ax.set_title("Posterior distribution\n(meta data)")
        sns.distplot(
            samples_posterior_meta[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])

    ax = axes[3]
    ax.set_title("Posterior distribution\n(test data)")
    sns.distplot(
        samples_posterior_test[site_name].squeeze(),
        kde_kws={"label": f"Test Task"},
        ax=ax,
    )
    if bm_param_idx is not None:
        ax.axvline(
            x=bm_test.get_task_by_index(0).param[bm_param_idx],
            color=sns.color_palette()[0],
        )

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[3].legend()
    axes[0].set_xlabel(site_name)
    axes[1].set_xlabel(site_name)
    axes[2].set_xlabel(site_name)
    axes[3].set_xlabel(site_name)
    fig.tight_layout()


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_x))
    y = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_y))
    for l, task in enumerate(bm):
        x[l, :] = task.x
        y[l, :] = task.y
    return x, y


def print_parameters():
    first = True
    for name in pyro.get_param_store().keys():
        if first:
            first = False
        else:
            print("\n")
        print(
            f"name  = {name}"
            f"\nshape = {pyro.param(name).shape}"
            f"\nvalue = {pyro.param(name)}"
        )


def freeze_parameters():
    for name, value in pyro.get_param_store().items():
        pyro.param[name] = pyro.param[name].detach()


def compute_regularizer(model):
    prior = model.get_prior_distribution()
    regularizer = torch.tensor(0.0)
    for prior_factor in prior:
        assert len(prior_factor.batch_shape) == 0
        normal = (
            dist.Normal(0.0, 1.0)
            .expand(prior_factor.event_shape)
            .to_event(len(prior_factor.event_shape))
        )
        kl = kl_divergence(prior_factor, normal)
        regularizer = regularizer + kl
    return regularizer


def train_model(model, guide, x, y, n_iter, initial_lr, final_lr=None):
    model.train()

    # optimizer
    optim_args = {}
    optim_args["lr"] = initial_lr
    if final_lr is not None:
        gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
        optim_args["lrd"] = gamma ** (1 / n_iter)
    optim = ClippedAdam(optim_args=optim_args)

    # SVI
    svi = SVI(model=model, guide=guide, optim=optim, loss=Trace_ELBO())

    # training loop
    pyro.clear_param_store()
    for i in range(n_iter):
        elbo = -svi.step(x=torch.tensor(x), y=torch.tensor(y))
        if i % 100 == 0 or i == len(range(n_iter)) - 1:
            print(f"[iter {i:04d}] elbo = {elbo:.4f}")

    model.eval()


def train_model_custom_loss(
    model, guide, x, y, n_iter, initial_lr, alpha_reg, final_lr=None
):
    model.train()

    # get parameters
    params_model = list(model.parameters())
    guide(x=torch.tensor(x), y=torch.tensor(y))
    params_guide = list(guide.parameters())
    params = params_model + params_guide

    ## optimizer
    # use the same tweaks as Pyro's ClippedAdam: LR decay and gradient clipping
    # (gradient clipping is implemented in the training loop itself)
    optim = AdamTorch(params=params, lr=initial_lr)
    if final_lr is not None:
        gamma = final_lr / initial_lr  # final learning rate will be gamma * initial_lr
        lr_decay = gamma ** (1 / n_iter)
        lr_scheduler = ExponentialLR(optimizer=optim, gamma=lr_decay)
    else:
        lr_scheduler = None

    # loss
    regularizer_fn = compute_regularizer
    elbo_fn = Trace_ELBO().differentiable_loss

    # training loop
    for i in range(n_iter):
        optim.zero_grad()
        regularizer = regularizer_fn(model=model)
        elbo = -elbo_fn(model=model, guide=guide, x=torch.tensor(x), y=torch.tensor(y))
        loss = -elbo + alpha_reg * regularizer
        loss.backward()
        clip_grad_norm_(params, max_norm=10.0)  # gradient clipping
        optim.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if i % 100 == 0 or i == len(range(n_iter)) - 1:
            print(f"[iter {i:04d}] elbo = {elbo:.4e} | reg = {regularizer:.4e}")

    model.eval()


def compute_log_likelihood(model, guide, x, y, n_samples):
    """
    Computes predictive log-likelihood using latent samples from guide using Predictive.
    """
    pyro.set_rng_seed(123)
    x, y = torch.tensor(x), torch.tensor(y)

    log_prob1 = 0.0
    for _ in range(n_samples):
        # run model with latent variables sampled from guide
        guide_trace = poutine.trace(guide).get_trace(x=x, y=y)
        replayed_model = poutine.replay(model, trace=guide_trace)
        model_trace1 = poutine.trace(replayed_model).get_trace(x=x, y=y)

        # https://www.programcreek.com/python/?CodeExample=log+likelihood
        obs_site = model_trace1.nodes["obs"]
        cur_log_prob1 = obs_site["fn"].log_prob(obs_site["value"])
        # TODO: this is wrong:
        log_prob1 += cur_log_prob1.sum()

    pyro.set_rng_seed(123)
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        parallel=True,
    )
    model_trace2 = predictive.get_vectorized_trace(x=x, y=y)

    # https://www.programcreek.com/python/?CodeExample=log+likelihood
    obs_site = model_trace2.nodes["obs"]
    log_prob2 = obs_site["fn"].log_prob(obs_site["value"])
    # TODO: this is wrong:
    log_prob2 = log_prob2.sum()

    return log_prob1 / n_samples, log_prob2 / n_samples


def main():
    # TODO: sample functions
    # TODO: use exact prior/posterior distributions, not the KDE (e.g., prior is task-independent!)
    # TODO: implement more complex priors (e.g., not factorized across layers?)

    ## flags, constants
    pyro.set_rng_seed(123)
    plot = True
    smoke_test = True
    # benchmark
    bm = Affine1D
    n_task_meta = 8
    n_points_per_task_meta = 16
    noise_stddev = 0.01
    # model
    n_hidden = 0
    d_hidden = 8
    infer_noise_stddev = True
    prior_type = "diagonal"
    # meta training
    do_meta_training = True
    n_iter_meta = 5000 if not smoke_test else 100
    initial_lr_meta = 0.1
    final_lr_meta = 0.00001
    alpha_reg_meta = 0.0
    # adaptation
    n_iter_test = 250
    initial_lr_test = 0.01
    final_lr_test = None
    # evaluation
    n_pred = 100
    n_samples = 1000
    max_plot_tasks = 5

    ## create benchmarks
    # meta benchmark
    bm_meta = bm(
        n_task=n_task_meta,
        n_datapoints_per_task=n_points_per_task_meta,
        output_noise=noise_stddev,
        seed_task=1234,
        seed_x=1235,
        seed_noise=1236,
    )
    x_meta, y_meta = collate_data(bm=bm_meta)
    x_pred_meta = np.linspace(-1.5, 1.5, n_pred)[None, :, None].repeat(
        n_task_meta, axis=0
    )
    # test benchmark
    bm_test = bm(
        n_task=1,
        n_datapoints_per_task=n_points_per_task_meta,
        output_noise=noise_stddev,
        seed_task=1235,
        seed_x=1236,
        seed_noise=1237,
    )
    x_test, y_test = collate_data(bm=bm_test)
    x_pred_test = np.linspace(-1.5, 1.5, n_pred)[None, :, None].repeat(1, axis=0)

    # create model
    mtbreg = MTBayesianRegression(
        d_x=bm_meta.d_x,
        d_y=bm_meta.d_y,
        n_hidden=n_hidden,
        d_hidden=d_hidden,
        noise_stddev=None if infer_noise_stddev else noise_stddev,
        prior_type=prior_type,
    )
    mtbreg.eval()

    prior_type = mtbreg.get_prior_distribution()

    ## obtain predictions before meta training
    with torch.no_grad():
        pred_summary_prior_meta_untrained, samples_prior_meta_untrained = predict(
            model=mtbreg, guide=None, x=x_pred_meta, n_samples=n_samples
        )

    ## print trace shapes
    print("\n********************")
    print("*** Trace shapes ***")
    print("********************")
    trace = poutine.trace(mtbreg).get_trace(x=torch.tensor(x_pred_meta))
    trace.compute_log_prob()
    print(trace.format_shapes())
    print("********************")

    ## print prior parameters
    print("\n************************")
    print("*** Prior parameters ***")
    print("************************")
    print_parameters()
    print("************************")

    ## do inference
    print("\n*******************************")
    print("*** Performing inference... ***")
    print("*******************************")
    # guide = AutoNormal(model=mtblr)
    guide_meta = AutoDiagonalNormal(model=mtbreg)
    # guide = AutoMultivariateNormal(model=mtblr)
    if do_meta_training:
        train_model_custom_loss(
            model=mtbreg,
            guide=guide_meta,
            x=x_meta,
            y=y_meta,
            n_iter=n_iter_meta,
            initial_lr=initial_lr_meta,
            final_lr=final_lr_meta,
            alpha_reg=alpha_reg_meta,
        )
    print("*******************************")

    ## print learned parameters
    print("\n****************************")
    print("*** Posterior parameters ***")
    print("****************************")
    print_parameters()
    print("****************************")

    ## obtain predictions after meta training
    # obtain prior predictions
    pred_summary_prior_meta_trained, samples_prior_meta_trained = predict(
        model=mtbreg, guide=None, x=x_pred_meta, n_samples=n_samples
    )
    # obtain posterior predictions
    pred_summary_posterior_meta, samples_posterior_meta = predict(
        model=mtbreg, guide=guide_meta, x=x_pred_meta, n_samples=n_samples
    )

    # ## freeze prior
    mtbreg.freeze_prior()

    # print freezed parameters
    print("\n**************************************")
    print("*** Posterior parameters (freezed) ***")
    print("**************************************")
    print_parameters()
    print("**************************************")

    ## do inference on test task
    print("\n*******************************")
    print("*** Performing inference on test task... ***")
    print("*******************************")
    # we need a new guide
    # guide_test = AutoNormal(model=mtblr)
    guide_test = AutoDiagonalNormal(model=mtbreg)
    # guide_test = AutoMultivariateNormal(model=mtblr)
    n_context = 1
    x_context, y_context = x_test[:, :n_context, :], y_test[:, :n_context, :]
    x_target, y_target = x_test[:, n_context:, :], y_test[:, n_context:, :]
    train_model(
        model=mtbreg,
        guide=guide_test,
        x=x_context,
        y=y_context,
        n_iter=n_iter_meta,
        initial_lr=initial_lr_meta,
        final_lr=final_lr_meta,
    )
    ll_target, ll_target2 = compute_log_likelihood(
        model=mtbreg,
        guide=guide_test,
        x=x_target,
        y=y_target,
        n_samples=10000,
    )
    ll_context, ll_context2 = compute_log_likelihood(
        model=mtbreg,
        guide=guide_test,
        x=x_context,
        y=y_context,
        n_samples=10000,
    )
    print(f" ll_target   = {ll_target:.2f}")
    print(f" ll_target2  = {ll_target2:.2f}")
    print(f" ll_context  = {ll_context:.2f}")
    print(f" ll_context2 = {ll_context2:.2f}")
    print("*******************************")
    # TODO: why do compute_log_likelihood and compute_log_likelihood2 produce different results?
    # TODO: make compute_log_likelihood2 standard and compute the correct log-likelihood with logsumexp!

    # print freezed parameters
    print("\n**************************************")
    print("*** Posterior parameters (freezed) ***")
    print("**************************************")
    print_parameters()
    print("**************************************")

    # obtain posterior predictions
    pred_summary_posterior_test, samples_posterior_test = predict(
        model=mtbreg, guide=guide_test, x=x_pred_test, n_samples=n_samples
    )

    # plot predictions
    if plot:
        plot_predictions(
            x_meta=x_meta,
            y_meta=y_meta,
            x_pred_meta=x_pred_meta,
            x_test_context=x_context,
            y_test_context=y_context,
            x_test_target=x_target,
            y_test_target=y_target,
            x_pred_test=x_pred_test,
            pred_summary_prior_meta_untrained=pred_summary_prior_meta_untrained,
            pred_summary_prior_meta_trained=pred_summary_prior_meta_trained,
            pred_summary_posterior_meta=pred_summary_posterior_meta,
            pred_summary_posterior_test=pred_summary_posterior_test,
            max_tasks=max_plot_tasks,
        )

        if n_hidden == 0:
            # plot prior and posterior distributions

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_distributions(
                    site_name="mtbnn.net.0.weight",
                    bm_meta=bm_meta,
                    bm_test=bm_test,
                    bm_param_idx=0
                    if isinstance(bm_meta, Linear1D) or isinstance(bm_meta, Affine1D)
                    else None,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posterior_test=samples_posterior_test,
                )

                plot_distributions(
                    site_name="mtbnn.net.0.bias",
                    bm_meta=bm_meta,
                    bm_test=bm_test,
                    bm_param_idx=1 if isinstance(bm_meta, Affine1D) else None,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posterior_test=samples_posterior_test,
                )

        plt.show()


if __name__ == "__main__":
    main()
