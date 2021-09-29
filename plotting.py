"""
Plotting functions for (multi-task) Bayesian neural networks.
"""
from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from util import split_tasks


def plot_predictions_for_one_set_of_tasks(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    pred_summary: dict,
    plot_obs: bool,
    ax: None,
    max_tasks: int,
    n_context: Optional[np.ndarray] = None,
):
    ## prepare data
    means_plt = pred_summary["_RETURN"]["mean"]
    means_perc5_plt = pred_summary["_RETURN"]["5%"]
    means_perc95_plt = pred_summary["_RETURN"]["95%"]
    obs_perc5_plt = pred_summary["obs"]["5%"]
    obs_perc95_plt = pred_summary["obs"]["95%"]

    # assert that all inputs have the same number of dimensions
    assert (
        x.ndim
        == y.ndim
        == x_pred.ndim
        == means_plt.ndim
        == means_perc5_plt.ndim
        == means_perc95_plt.ndim
        == obs_perc5_plt.ndim
        == obs_perc95_plt.ndim
    )
    # check that all inputs are one-dimensional
    assert (
        x.shape[-1]
        == y.shape[-1]
        == x_pred.shape[-1]
        == means_plt.shape[-1]
        == means_perc5_plt.shape[-1]
        == means_perc95_plt.shape[-1]
        == obs_perc5_plt.shape[-1]
        == obs_perc95_plt.shape[-1]
        == 1
    )
    # check that all inputs have the same number of tasks
    assert (
        x.shape[0]
        == y.shape[0]
        == means_plt.shape[0]
        == means_perc5_plt.shape[0]
        == means_perc95_plt.shape[0]
        == obs_perc5_plt.shape[0]
        == obs_perc95_plt.shape[0]
    )
    # squeeze data dimension
    (
        x,
        y,
        x_pred,
        means_plt,
        means_perc5_plt,
        means_perc95_plt,
        obs_perc5_plt,
        obs_perc95_plt,
    ) = (
        x.squeeze(-1),
        y.squeeze(-1),
        x_pred.squeeze(-1),
        means_plt.squeeze(-1),
        means_perc5_plt.squeeze(-1),
        means_perc95_plt.squeeze(-1),
        obs_perc5_plt.squeeze(-1),
        obs_perc95_plt.squeeze(-1),
    )

    ## prepare plot
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ## plot
    n_task = x.shape[0]
    for l in range(n_task):
        if l == max_tasks:
            break
        line = ax.plot(x_pred[l, :], means_plt[l, :])[0]
        ax.scatter(x[l, :], y[l, :], color=line.get_color())
        if n_context is not None:
            x_context, y_context, _, _ = split_tasks(
                x=x[l : l + 1, :][:, :, None],
                y=y[l : l + 1, :][:, :, None],
                n_context=n_context,
            )
            ax.scatter(
                x_context,
                y_context,
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
    x_test,
    y_test,
    n_contexts_test,
    x_pred_test,
    pred_summary_prior_meta_untrained,
    pred_summary_prior_meta_trained,
    pred_summary_posterior_meta,
    pred_summaries_posterior_test,
    max_tasks=3,
):
    fig, axes = plt.subplots(
        nrows=2, ncols=3 + len(n_contexts_test), figsize=(16, 8), sharey=True
    )
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

    for i, n_context in enumerate(n_contexts_test):
        ax = axes[0, 3 + i]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Posterior Mean\n(test data, n_ctx={n_context:d})")
        plot_predictions_for_one_set_of_tasks(
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            n_context=n_context,
            pred_summary=pred_summaries_posterior_test[i],
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

    for i, n_context in enumerate(n_contexts_test):
        ax = axes[1, 3 + i]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Posterior Observation\n(test data, n_ctx={n_context:d})")
        plot_predictions_for_one_set_of_tasks(
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            n_context=n_context,
            pred_summary=pred_summaries_posterior_test[i],
            ax=ax,
            plot_obs=True,
            max_tasks=max_tasks,
        )
    fig.tight_layout()

    return fig


def plot_distributions(
    site_name,
    bm_meta_params,
    bm_test_params,
    samples_prior_meta_untrained,
    samples_prior_meta_trained,
    samples_posterior_meta,
    samples_posteriors_test,
    n_contexts_test,
    max_tasks=3,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3 + len(n_contexts_test),
        figsize=(16, 8),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(f"Prior and Posterior Distributions of site '{site_name}'")

    n_meta_tasks = samples_prior_meta_untrained[site_name].shape[1]
    n_test_tasks = samples_posteriors_test[0][site_name].shape[1]

    for l in range(n_meta_tasks):
        if l == max_tasks:
            break
        ax = axes[0, 0]
        ax.set_title("Prior distribution\n(untrained)")
        sns.distplot(
            samples_prior_meta_untrained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

        ax = axes[0, 1]
        ax.set_title("Prior distribution\n(trained on meta data)")
        sns.distplot(
            samples_prior_meta_trained[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

        ax = axes[0, 2]
        ax.set_title("Posterior distribution\n(meta data)")
        sns.distplot(
            samples_posterior_meta[site_name].squeeze()[:, l],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

    for i, n_context in enumerate(n_contexts_test):
        for l in range(n_test_tasks):
            if l == max_tasks:
                break
            ax = axes[0, 3 + i]
            ax.set_title(f"Posterior distribution\n(test data, n_ctx={n_context:d})")
            sns.distplot(
                samples_posteriors_test[i][site_name].squeeze()[:, l],
                kde_kws={"label": f"Test Task {l}"},
                ax=ax,
            )
            if bm_test_params is not None:
                ax.axvline(x=bm_test_params[l], color=sns.color_palette()[l])

    for ax in axes[0]:
        ax.legend()
        ax.set_xlabel(site_name)
    fig.tight_layout()

    return fig


def plot_metrics(
    learning_curve_meta, learning_curves_test, lls, lls_context, n_contexts
):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,8), squeeze=False)
    fig.suptitle("Metrics")

    ax = axes[0, 0]
    ax.set_title("Learning Curve (meta training)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog")  # TODO: think about symlog scaling
    if learning_curve_meta is not None:
        ax.plot(
            np.arange(len(learning_curve_meta)),
            learning_curve_meta,
        )
        ax.legend()
    ax.grid()

    ax = axes[0, 1]
    ax.set_title("Learning Curve (adaptation)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog")  # TODO: think about symlog scaling
    for learning_curve, n_context in zip(learning_curves_test, n_contexts):
        ax.plot(
            np.arange(len(learning_curve)),
            learning_curve,
            label=f"n_ctx={n_context:d}",
        )
    ax.legend()
    ax.grid()

    ax = axes[0, 2]
    ax.set_title("Marginal Log-Likelihood")
    ax.set_xlabel("n_context")
    ax.set_xticks(n_contexts)
    ax.set_ylabel("marginal ll")
    ax.plot(n_contexts, lls, label="all data")
    ax.plot(n_contexts, lls_context, label="context only")
    ax.legend()
    ax.grid()

    fig.tight_layout()
    
    return fig
