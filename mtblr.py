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
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.optim import Adam


def check_shapes(x, y):
    assert x.ndim == 3  # (n_task, n_points, d_x)
    assert x.shape[-1] == 1  # d_x = 1
    if y is not None:
        assert x.shape == y.shape  # d_y = 1


def mtblr_model(x, y=None):
    check_shapes(x=x, y=y)

    sigma = pyro.sample("sigma", dist.Uniform(0.0, 2.0))
    with pyro.plate("tasks", x.shape[0], dim=-3):
        m = pyro.sample("m", dist.Normal(0.0, 1.0))
        mean = m * x
        with pyro.plate("data", x.shape[1], dim=-2):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
    return mean


def mtblr_guide(x, y=None):
    check_shapes(x=x, y=y)

    sigma_loc = pyro.param(
        "sigma_loc", torch.tensor(1.0), constraint=constraints.positive
    )
    pyro.sample("sigma", dist.Uniform(max(sigma_loc - 0.05, 0.0), sigma_loc + 0.05))
    with pyro.plate("tasks", x.shape[0], dim=-3):
        m_loc = pyro.param("m_loc", torch.randn((x.shape[0], 1, 1)))
        m_scale = pyro.param(
            "m_scale", torch.ones((x.shape[0], 1, 1)), constraint=constraints.positive
        )
        pyro.sample("m", dist.Normal(m_loc, m_scale))


def predict(model, guide, x: np.ndarray, n_samples: int):
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
        base_line = ax.plot(x_pred[l], means_plt[l])[0]
        ax.scatter(x_train[l], y_train[l], color=base_line.get_color())
        if not plot_obs:
            ax.fill_between(
                x_pred[l],
                means_perc5_plt[l],
                means_perc95_plt[l],
                alpha=0.3,
                color=base_line.get_color(),
            )
        else:
            ax.fill_between(
                x_pred[l],
                obs_perc5_plt[l],
                obs_perc95_plt[l],
                alpha=0.3,
                color=base_line.get_color(),
            )


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_x))
    y = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_y))
    for l, task in enumerate(bm):
        x[l] = task.x
        y[l] = task.y
    return x, y


if __name__ == "__main__":
    # flags, constants
    plot = True
    smoke_test = False
    n_task = 4
    n_points_per_task = 8
    output_noise = 0.1
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

    # now do inference
    adam = Adam({"lr": 0.05})
    svi = SVI(model=mtblr_model, guide=mtblr_guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for i in range(n_iter):
        elbo = svi.step(x=torch.tensor(x), y=torch.tensor(y))
        if i % 10 == 0:
            print(f"[iter {i:04d}] elbo = {elbo:.4f}")

    # print learned parameters
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))

    # obtain prior predictions
    n_pred = 100
    x_pred = np.linspace(-1.0, 1.0, n_pred)[None, :, None].repeat(n_task, axis=0)
    pred_summary_prior, samples_prior = predict(
        model=mtblr_model, guide=None, x=x_pred, n_samples=n_samples
    )

    # obtain posterior predictions
    pred_summary_posterior, samples_posterior = predict(
        model=mtblr_model, guide=mtblr_guide, x=x_pred, n_samples=n_samples
    )

    # plot predictions
    if plot:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharey=True)
        fig.suptitle("Prior and Posterior Predictions")

        ax = axes[0, 0]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Prior Mean")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_prior,
            ax=ax,
            plot_obs=False,
        )

        ax = axes[0, 1]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Posterior Mean")
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
        ax.set_title("Prior Observation")
        plot_predictions(
            x_train=x,
            y_train=y,
            x_pred=x_pred,
            pred_summary=pred_summary_prior,
            ax=ax,
            plot_obs=True,
        )

        ax = axes[1, 1]
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
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True)
    fig.suptitle("Prior and Posterior Distributions")

    for l in range(n_task):
        ax = axes[0]
        sns.distplot(samples_prior["m"][:, l], kde_kws={"label": f"Task {l}"}, ax=ax)
        ax = axes[1]
        sns.distplot(
            samples_posterior["m"][:, l], kde_kws={"label": f"Task {l}"}, ax=ax
        )
    axes[0].legend()
    axes[1].legend()
    plt.show()
