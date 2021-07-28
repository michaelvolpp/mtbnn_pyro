from typing import Optional

import numpy as np
import pandas as pd
import pyro
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from metalearning_benchmarks import Linear1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from numpy.core.fromnumeric import sort
from pyro import distributions as dist
from pyro.distributions import constraints
from pyro.infer import SVI, Predictive, Trace_ELBO, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import Adam
from torch import nn


def check_shapes(x, y):
    assert x.ndim == 3  # (n_points, n_tasks, d_x)
    assert x.shape[-1] == 1  # d_x = 1
    if y is not None:
        assert x.shape == y.shape  # d_y = 1


class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        if bias:
            raise NotImplementedError

        self.m_loc_prior = pyro.param("m_loc_prior", torch.tensor(0.0))
        self.m_scale_prior = pyro.param(
            "m_scale_prior",
            torch.tensor(1.0),
            constraint=constraints.positive,
        )
        self.m = PyroSample(
            dist.Normal(self.m_loc_prior, self.m_scale_prior)
            .expand([in_features, out_features])
            .to_event(2)
        )

    def forward(self, x):
        assert self.m.ndim == x.ndim == 3
        y = torch.einsum("lyx,nlx->nly", self.m, x)
        return y

class MTBLR(PyroModule):
    def __init__(self, d_x: int, d_y: int):
        super().__init__()
        self.linear = BayesianLinear(in_features=d_x, out_features=d_y, bias=False)

    def forward(self, x: torch.tensor, d_y: int, y: Optional[torch.tensor] = None):
        check_shapes(x=x, y=y)
        n_points = x.shape[0]
        n_tasks = x.shape[1]
        d_x = x.shape[2]
        if y is not None:
            assert y.shape == (n_points, n_tasks, d_y)

        sigma = pyro.sample("sigma", dist.Uniform(0.0, 10.0))
        with pyro.plate("tasks", n_tasks):
            mean = self.linear(x)
            assert mean.shape == (n_points, n_tasks, d_y)
            with pyro.plate("data", n_points):
                obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
                assert obs.shape == (n_points, n_tasks, d_y)
        return mean


def mtblr_guide(x: torch.tensor, d_y: int, y: Optional[torch.tensor] = None):
    check_shapes(x=x, y=y)
    n_points = x.shape[0]
    n_tasks = x.shape[1]
    d_x = x.shape[2]
    if y is not None:
        assert y.shape == (n_points, n_tasks, d_y)

    sigma_scale = 1e-4
    sigma_loc = pyro.param(
        "sigma_loc", torch.tensor(1.0), constraint=constraints.positive
    )
    pyro.sample("sigma", dist.Normal(max(sigma_loc, 5 * sigma_scale), sigma_scale))
    with pyro.plate("tasks", n_tasks):
        m_loc = pyro.param("m_loc", torch.randn((n_tasks, d_y, d_x)))
        m_scale = pyro.param(
            "m_scale", torch.ones((n_tasks, d_y, d_x)), constraint=constraints.positive
        )
        m = pyro.sample("m", dist.Normal(m_loc, m_scale).to_event(2))
        assert m.shape == (n_tasks, d_y, d_x)


def predict(model, guide, x: np.ndarray, d_y: int, n_samples: int):
    # TODO: understand shapes of svi_samples
    predictive = Predictive(
        model=model,
        guide=guide,
        num_samples=n_samples,
        return_sites=(
            "m",
            "obs",
            "sigma",
            "_RETURN",
        ),
    )
    svi_samples = predictive(x=torch.tensor(x), d_y=d_y, y=None)
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


def plot_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    pred_summary: dict,
    plot_obs: bool,
    ax: None,
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
        base_line = ax.plot(x_pred[:, l], means_plt[:, l])[0]
        ax.scatter(x_train[:, l], y_train[:, l], color=base_line.get_color())
        if not plot_obs:
            ax.fill_between(
                x_pred[:, l],
                means_perc5_plt[:, l],
                means_perc95_plt[:, l],
                alpha=0.3,
                color=base_line.get_color(),
            )
        else:
            ax.fill_between(
                x_pred[:, l],
                obs_perc5_plt[:, l],
                obs_perc95_plt[:, l],
                alpha=0.3,
                color=base_line.get_color(),
            )


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_points_per_task, bm.n_task, bm.d_x))
    y = np.zeros((bm.n_points_per_task, bm.n_task, bm.d_y))
    for l, task in enumerate(bm):
        x[:, l] = task.x
        y[:, l] = task.y
    return x, y


if __name__ == "__main__":
    # seed
    pyro.set_rng_seed(123)

    # flags, constants
    plot = True
    smoke_test = False
    n_task = 6
    n_points_per_task = 16
    output_noise = 0.35
    n_pred = 100
    x_pred = np.linspace(-1.0, 1.0, n_pred)[:, None, None].repeat(n_task, axis=1)
    n_iter = 1000 if not smoke_test else 10
    n_samples = 1000 if not smoke_test else 10

    # create benchmark
    bm = Linear1D(
        n_task=n_task,
        n_datapoints_per_task=n_points_per_task,
        output_noise=output_noise,
        seed_task=1234,
        seed_x=1235,
        seed_noise=1236,
    )
    x, y = collate_data(bm=bm)

    # create model
    mtblr = MTBLR(d_x=bm.d_x, d_y=bm.d_y)

    # obtain predictions before training
    mtblr.eval()
    pred_summary_prior_untrained, samples_prior_untrained = predict(
        model=mtblr, guide=None, x=x_pred, d_y=bm.d_y, n_samples=n_samples
    )

    # now do inference
    mtblr.train()
    adam = Adam({"lr": 0.05})
    guide = AutoDiagonalNormal(model=mtblr) 
    svi = SVI(model=mtblr, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for i in range(n_iter):
        elbo = svi.step(x=torch.tensor(x), d_y=bm.d_y, y=torch.tensor(y))
        if i % 10 == 0:
            print(f"[iter {i:04d}] elbo = {elbo:.4f}")

    # print learned parameters
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    # obtain prior predictions
    pred_summary_prior, samples_prior = predict(
        model=mtblr, guide=None, x=x_pred, d_y=bm.d_y, n_samples=n_samples
    )

    # obtain posterior predictions
    pred_summary_posterior, samples_posterior = predict(
        model=mtblr, guide=guide, x=x_pred, d_y=bm.d_y, n_samples=n_samples
    )

    # plot predictions
    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), sharey=True)
        fig.suptitle("Prior and Posterior Predictions")

        ax = axes[0, 0]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Prior Mean (untrained)")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_prior_untrained,
            ax=ax,
            plot_obs=False,
        )

        ax = axes[0, 1]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Posterior Mean (trained)")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_prior,
            ax=ax,
            plot_obs=False,
        )

        ax = axes[0, 2]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Posterior Mean (untrained)")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_posterior,
            ax=ax,
            plot_obs=False,
        )

        ax = axes[1, 0]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Prior Observation (untrained)")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_prior_untrained,
            ax=ax,
            plot_obs=True,
        )

        ax = axes[1, 1]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Prior Observation (trained)")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_prior,
            ax=ax,
            plot_obs=True,
        )

        ax = axes[1, 2]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Posterior Observation")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_posterior,
            ax=ax,
            plot_obs=True,
        )

    # plot prior and posterior
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6), sharex=True)
    fig.suptitle("Prior and Posterior Distributions")

    for l, task in enumerate(bm):
        ax = axes[0]
        ax.set_title("Prior distribution (untrained)")
        sns.distplot(
            samples_prior_untrained["m"].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        ax.axvline(x=task.param[0], color=sns.color_palette()[l])

        ax = axes[1]
        ax.set_title("Prior distribution (trained)")
        sns.distplot(
            samples_prior["m"].squeeze()[:, l], kde_kws={"label": f"Task {l}"}, ax=ax
        )
        ax.axvline(x=task.param[0], color=sns.color_palette()[l])

        ax = axes[2]
        ax.set_title("Posterior distribution")
        sns.distplot(
            samples_posterior["m"].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        ax.axvline(x=task.param[0], color=sns.color_palette()[l])
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_xlabel("m")
    axes[1].set_xlabel("m")
    axes[2].set_xlabel("m")
    plt.show()
