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
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal, AutoMultivariateNormal
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.optim import Adam, ClippedAdam
from torch import nn

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
        prior: str = "isotropic",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        ## weight prior
        if prior == "isotropic":
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
        elif prior == "diagonal":
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
            raise ValueError(f"Unknown prior specification '{prior}'!")
        self.weight = PyroSample(self.weight_prior)

        ## bias prior
        if bias:
            if prior == "isotropic":
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
            elif prior == "diagonal":
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
                raise ValueError(f"Unknown prior specification '{prior}'!")
            self.bias = PyroSample(self.bias_prior)

    def forward(self, x):
        # check shapes
        # x.shape = (n_tasks, n_points, d_x)
        assert x.ndim == 3
        n_tasks = x.shape[0]
        # self.weight.batch_shape = (n_tasks, 1)
        # self.weight.event_shape = (self.out_features, self.in_features)
        assert self.weight.shape == (n_tasks, 1, self.out_features, self.in_features)
        # self.bias.batch_shape = (n_tasks, 1)
        # self.bias.event_shape = (self.out_features)
        assert self.bias.shape == (n_tasks, 1, self.out_features)

        # TODO: why do we need x.float() here as soon as we have more than zero layers?
        y = torch.einsum("lyx,lnx->lny", self.weight.squeeze(1), x.float())
        y = y + self.bias.squeeze(1)[:, None, :]
        return y

    def freeze_parameters(self):
        # self.weight_loc_prior = self.weight_loc_prior.clone().detach()
        self.weight_prior_loc = torch.tensor(0.0, requires_grad=False)
        # self.weight_loc_prior = 0.0


class MTBNN(PyroModule):
    def __init__(
        self, in_features: int, out_features: int, n_hidden: int, d_hidden: int
    ):
        super().__init__()

        if n_hidden == 1:
            modules = []
            modules.append(
                MTBayesianLinear(
                    in_features=in_features, out_features=d_hidden, bias=True
                )
            )
            modules.append(PyroModule[nn.Tanh]())
            modules.append(
                MTBayesianLinear(
                    in_features=d_hidden, out_features=out_features, bias=True
                )
            )
            self.net = PyroModule[nn.Sequential](*modules)
        elif n_hidden == 0:
            self.net = MTBayesianLinear(
                in_features=in_features, out_features=out_features, bias=True
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.net(x)


class MTBayesianRegression(PyroModule):
    def __init__(
        self,
        d_x: int,
        d_y: int,
        n_hidden: int,
        d_hidden: int,
        noise_stddev: Optional[float] = None,
    ):
        super().__init__()

        ## multi-task BNN
        self.mtbnn = MTBNN(
            in_features=d_x, out_features=d_y, n_hidden=n_hidden, d_hidden=d_hidden
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
            with pyro.plate("data", n_points, dim=-1):
                obs = pyro.sample(
                    "obs", dist.Normal(mean, noise_stddev).to_event(1), obs=y
                )  # score mean predictions against ground truth
        return mean


def predict(model, guide, x: np.ndarray, n_samples: int):
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        return_sites=(
            # if model is linear, those sites are available
            "mtbnn.net.weight",
            "mtbnn.net.bias",
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


def plot_one_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    pred_summary: dict,
    plot_obs: bool,
    ax: None,
    max_tasks: int,
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
    for l in range(n_task):
        if l == max_tasks:
            break
        base_line = ax.plot(x_pred[l, :], means_plt[l, :])[0]
        ax.scatter(x_train[l, :], y_train[l, :], color=base_line.get_color())
        if not plot_obs:
            ax.fill_between(
                x_pred[l, :],
                means_perc5_plt[l, :],
                means_perc95_plt[l, :],
                alpha=0.3,
                color=base_line.get_color(),
            )
        else:
            ax.fill_between(
                x_pred[l, :],
                obs_perc5_plt[l, :],
                obs_perc95_plt[l, :],
                alpha=0.3,
                color=base_line.get_color(),
            )
    ax.grid()


def plot_predictions(
    x_train,
    y_train,
    x_pred,
    pred_summary_prior_untrained,
    pred_summary_prior,
    pred_summary_posterior,
    max_tasks=3,
):
    assert x_train.shape[-1] == y_train.shape[-1] == x_pred.shape[-1]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharey=True)
    fig.suptitle(f"Prior and Posterior Predictions")

    ax = axes[0, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean (untrained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior_untrained,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Mean (trained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[0, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Mean")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_posterior,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    ax = axes[1, 0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation (untrained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior_untrained,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Prior Observation (trained)")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_prior,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    ax = axes[1, 2]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Observation")
    plot_one_prediction(
        x_train=x_train,
        y_train=y_train,
        x_pred=x_pred,
        pred_summary=pred_summary_posterior,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )
    fig.tight_layout()


def plot_distributions(
    site_name,
    bm,
    bm_param_idx,
    samples_prior_untrained,
    samples_prior,
    samples_posterior,
    max_tasks=3,
):
    # TODO: why do we need to squeeze the slope samples?
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharex=True)
    fig.suptitle(f"Prior and Posterior Distributions of site '{site_name}'")

    for l, task in enumerate(bm):
        if l == max_tasks:
            break
        ax = axes[0]
        ax.set_title("Prior distribution (untrained)")
        sns.distplot(
            samples_prior_untrained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])

        ax = axes[1]
        ax.set_title("Prior distribution (trained)")
        sns.distplot(
            samples_prior[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])

        ax = axes[2]
        ax.set_title("Posterior distribution")
        sns.distplot(
            samples_posterior[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_param_idx is not None:
            ax.axvline(x=task.param[bm_param_idx], color=sns.color_palette()[l])
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_xlabel(site_name)
    axes[1].set_xlabel(site_name)
    axes[2].set_xlabel(site_name)
    fig.tight_layout()


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_x))
    y = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_y))
    for l, task in enumerate(bm):
        x[l, :] = task.x
        y[l, :] = task.y
    return x, y


def print_parameters():
    for name in pyro.get_param_store().keys():
        print(
            f"\n\nname  = {name}"
            f"\nshape = {pyro.param(name).shape}"
            f"\nvalue = {pyro.param(name)}"
        )


def freeze_parameters():
    for name, value in pyro.get_param_store().items():
        pyro.param[name] = pyro.param[name].detach()


if __name__ == "__main__":
    # TODO: implement adaptation
    # TODO: use more expressive prior
    # TODO: multivariate normal guide
    # seed
    pyro.set_rng_seed(123)

    ## flags, constants
    plot = True
    smoke_test = False
    # benchmark
    n_task = 16
    n_points_per_task = 32
    noise_stddev = 0.01
    # model
    n_hidden = 1
    d_hidden = 4
    infer_noise_stddev = False
    # training
    n_iter = 5000 if not smoke_test else 1000
    initial_lr = 0.1
    gamma = 0.00001  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / n_iter)
    adam = ClippedAdam({"lr": initial_lr, "lrd": lrd})
    # evaluation
    n_pred = 100
    x_pred = np.linspace(-1.5, 1.5, n_pred)[None, :, None].repeat(n_task, axis=0)
    n_samples = 1000
    max_plot_tasks = 5

    # create benchmark
    bm = Affine1D(
        n_task=n_task,
        n_datapoints_per_task=n_points_per_task,
        output_noise=noise_stddev,
        seed_task=1234,
        seed_x=1235,
        seed_noise=1236,
    )
    x, y = collate_data(bm=bm)

    # create model
    mtblr = MTBayesianRegression(
        d_x=bm.d_x,
        d_y=bm.d_y,
        n_hidden=n_hidden,
        d_hidden=d_hidden,
        noise_stddev=None if infer_noise_stddev else noise_stddev,
    )

    ## obtain predictions before training
    mtblr.eval()
    with torch.no_grad():
        pred_summary_prior_untrained, samples_prior_untrained = predict(
            model=mtblr, guide=None, x=x_pred, n_samples=n_samples
        )

    ## print trace shapes
    trace = poutine.trace(mtblr).get_trace(x=torch.tensor(x_pred))
    trace.compute_log_prob()
    print(trace.format_shapes())

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
    mtblr.train()
    # guide = AutoDiagonalNormal(model=mtblr)
    guide = AutoDiagonalNormal(model=mtblr)
    svi = SVI(model=mtblr, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for i in range(n_iter):
        elbo = svi.step(x=torch.tensor(x), y=torch.tensor(y))
        if i % 100 == 0:
            print(f"[iter {i:04d}] elbo = {elbo:.4f}")
    print("*******************************")

    ## print learned parameters
    print("\n****************************")
    print("*** Posterior parameters ***")
    print("****************************")
    print_parameters()
    print("****************************")

    ## obtain predictions after training
    mtblr.eval()
    # obtain prior predictions
    pred_summary_prior, samples_prior = predict(
        model=mtblr, guide=None, x=x_pred, n_samples=n_samples
    )

    # obtain posterior predictions
    pred_summary_posterior, samples_posterior = predict(
        model=mtblr, guide=guide, x=x_pred, n_samples=n_samples
    )

    # plot predictions
    if plot:
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary_prior_untrained=pred_summary_prior_untrained,
            pred_summary_prior=pred_summary_prior,
            pred_summary_posterior=pred_summary_posterior,
            max_tasks=max_plot_tasks,
        )

        if n_hidden == 0:
            # plot prior and posterior distributions

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_distributions(
                    site_name="mtbnn.net.weight",
                    bm=bm,
                    bm_param_idx=0
                    if isinstance(bm, Linear1D) or isinstance(bm, Affine1D)
                    else None,
                    samples_prior_untrained=samples_prior_untrained,
                    samples_prior=samples_prior,
                    samples_posterior=samples_posterior,
                )

                plot_distributions(
                    site_name="mtbnn.net.bias",
                    bm=bm,
                    bm_param_idx=1 if isinstance(bm, Affine1D) else None,
                    samples_prior_untrained=samples_prior_untrained,
                    samples_prior=samples_prior,
                    samples_posterior=samples_posterior,
                )

        plt.show()
