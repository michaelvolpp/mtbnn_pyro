"""
Plotting functions for (multi-task) Bayesian neural networks.
"""
from typing import List, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mtutils.mtutils import split_tasks


def plot_predictions_for_one_set_of_tasks(
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
    plot_obs: bool,
    ax: None,
    max_tasks: int,
    pred_summary: Optional[dict] = None,
    samples: Optional[dict] = None,
    max_samples: Optional[int] = 100,
    n_context: Optional[np.ndarray] = None,
):
    assert not (pred_summary is None and samples is None)

    ## prepare data
    if pred_summary is not None:
        means_plt = pred_summary["_RETURN"]["mean"]
        means_perc5_plt = pred_summary["_RETURN"]["5%"]
        means_perc95_plt = pred_summary["_RETURN"]["95%"]
        obs_perc5_plt = pred_summary["obs"]["5%"]
        obs_perc95_plt = pred_summary["obs"]["95%"]
    if samples is not None:
        samples_means_plt = samples["_RETURN"]
        samples_obs_plt = samples["obs"]

    # assert that all inputs have the same number of dimensions
    if pred_summary is not None:
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
    if samples is not None:
        assert x.ndim == samples_means_plt.ndim - 1 == samples_obs_plt.ndim - 1
    # check that all inputs are one-dimensional
    if pred_summary is not None:
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
    if samples is not None:
        assert x.shape[-1] == samples_means_plt.shape[-1] == samples_obs_plt.shape[-1]
    # check that all inputs have the same number of tasks
    if pred_summary is not None:
        assert (
            x.shape[0]
            == y.shape[0]
            == means_plt.shape[0]
            == means_perc5_plt.shape[0]
            == means_perc95_plt.shape[0]
            == obs_perc5_plt.shape[0]
            == obs_perc95_plt.shape[0]
        )
    if samples is not None:
        assert x.shape[0] == samples_means_plt.shape[1] == samples_obs_plt.shape[1]
    # squeeze data dimension
    if pred_summary is not None:
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
    if samples is not None:
        samples_means_plt, samples_obs_plt = (
            samples_means_plt.squeeze(-1),
            samples_obs_plt.squeeze(-1),
        )

    ## prepare plot
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ## plot
    n_task = x.shape[0]
    n_samples = samples_means_plt.shape[0] if samples is not None else None
    for l in range(n_task):
        if l == max_tasks:
            break
        sc = ax.scatter(x[l, :], y[l, :], s=10)
        color = sc.get_facecolors()[0].tolist()
        if pred_summary is not None:
            ax.plot(x_pred[l, :], means_plt[l, :])[0]
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
                color=color,
            )
        if not plot_obs:
            if pred_summary is not None:
                ax.fill_between(
                    x_pred[l, :],
                    means_perc5_plt[l, :],
                    means_perc95_plt[l, :],
                    alpha=0.3,
                    color=color,
                )
            if samples is not None:
                for s in range(n_samples):
                    if s == max_samples:
                        break
                    ax.plot(
                        x_pred[l, :],
                        samples_means_plt[s, l],
                        alpha=0.1,
                        color=color,
                    )
        else:
            if pred_summary is not None:
                ax.fill_between(
                    x_pred[l, :],
                    obs_perc5_plt[l, :],
                    obs_perc95_plt[l, :],
                    alpha=0.3,
                    color=color,
                )
            if samples is not None:
                for s in range(max_samples):
                    if s == max_samples:
                        break
                    ax.plot(
                        x_pred[l, :],
                        samples_obs_plt[s, l],
                        alpha=0.1,
                        color=color,
                    )

    ax.grid()


def plot_predictions(
    x_meta,
    y_meta,
    x_pred_meta,
    pred_summary_prior_meta_untrained,
    pred_summary_prior_meta_trained,
    pred_summary_posterior_meta,
    x_test,
    y_test,
    n_contexts_test,
    x_pred_test,
    pred_summaries_posterior_test,
    n_contexts_plot,
    max_tasks,
    samples_prior_meta_untrained = None,
    samples_prior_meta_trained = None,
):
    n_meta_plots = 3 if pred_summary_posterior_meta is not None else 2
    n_test_plots = len(n_contexts_plot) if n_contexts_plot is not None else 0
    fig, axes = plt.subplots(
        nrows=2, ncols=n_meta_plots + n_test_plots, figsize=(16, 8), sharey=True
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
        samples=samples_prior_meta_untrained,
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
        samples=samples_prior_meta_trained,
        ax=ax,
        plot_obs=False,
        max_tasks=max_tasks,
    )

    if pred_summary_posterior_meta is not None:
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

    if x_test is not None:
        plt_ct = -1
        for i, n_context in enumerate(n_contexts_test):
            if n_context not in n_contexts_plot:
                continue

            plt_ct += 1
            ax = axes[0, 3 + plt_ct]
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
        samples=samples_prior_meta_untrained,
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
        samples=samples_prior_meta_trained,
        ax=ax,
        plot_obs=True,
        max_tasks=max_tasks,
    )

    if pred_summary_posterior_meta is not None:
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

    if x_test is not None:
        plt_ct = -1
        for i, n_context in enumerate(n_contexts_test):
            if n_context not in n_contexts_plot:
                continue

            plt_ct += 1
            ax = axes[1, 3 + plt_ct]
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
    site_idx,
    bm_meta_params,
    bm_test_params,
    samples_prior_meta_untrained,
    samples_prior_meta_trained,
    samples_posterior_meta,
    samples_posteriors_test,
    n_contexts_test,
    max_tasks,
    n_contexts_plot,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3 + len(n_contexts_plot),
        figsize=(16, 8),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Prior and Posterior Distributions of site '{site_name}[...,{site_idx:d}]'"
    )

    n_meta_tasks = samples_prior_meta_untrained[site_name].shape[1]
    n_test_tasks = samples_posteriors_test[0][site_name].shape[1]

    for l in range(n_meta_tasks):
        if l == max_tasks:
            break
        ax = axes[0, 0]
        ax.set_title("Prior distribution\n(untrained)")
        sns.distplot(
            samples_prior_meta_untrained[site_name].squeeze()[:, l, site_idx],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

        ax = axes[0, 1]
        ax.set_title("Prior distribution\n(trained on meta data)")
        sns.distplot(
            samples_prior_meta_trained[site_name].squeeze()[:, l, site_idx],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

        ax = axes[0, 2]
        ax.set_title("Posterior distribution\n(meta data)")
        sns.distplot(
            samples_posterior_meta[site_name].squeeze()[:, l, site_idx],
            kde_kws={"label": f"Task {l}"},
            ax=ax,
        )
        if bm_meta_params is not None:
            ax.axvline(x=bm_meta_params[l], color=sns.color_palette()[l])

    plt_ct = -1
    for i, n_context in enumerate(n_contexts_test):
        if n_context not in n_contexts_plot:
            continue

        plt_ct += 1
        for l in range(n_test_tasks):
            if l == max_tasks:
                break
            ax = axes[0, 3 + plt_ct]
            ax.set_title(f"Posterior distribution\n(test data, n_ctx={n_context:d})")
            sns.distplot(
                samples_posteriors_test[i][site_name].squeeze()[:, l, site_idx],
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
    learning_curve_meta,
    learning_curves_test=None,
    rmses=None,
    rmses_context=None,
    lls=None,
    lls_context=None,
    n_contexts=None,
):
    n_plots = 1
    if learning_curves_test is not None:
        n_plots += 1
    if rmses is not None:
        assert rmses_context is not None
        n_plots += 1
    if lls is not None:
        assert lls_context is not None
        n_plots += 1
    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=(16, 8), squeeze=False)
    fig.suptitle("Metrics")

    ax_ct = 0
    ax = axes[0, ax_ct]
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
    ax_ct += 1

    if learning_curves_test is not None:
        ax = axes[0, ax_ct]
        ax.set_title("Learning Curves (adaptation)")
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
        ax_ct += 1

    if lls is not None:
        ax = axes[0, ax_ct]
        ax.set_title("Marginal Log-Likelihood")
        ax.set_xlabel("n_context")
        ax.set_xticks(n_contexts)
        ax.set_ylabel("marginal ll")
        ax.plot(n_contexts, lls, label="all data")
        ax.plot(n_contexts, lls_context, label="context only")
        ax.legend()
        ax.grid()
        ax_ct += 1

    if rmses is not None:
        ax = axes[0, ax_ct]
        ax.set_title("Root Mean Squared Errors")
        ax.set_xlabel("n_context")
        ax.set_xticks(n_contexts)
        ax.set_ylabel("RMSE")
        ax.plot(n_contexts, rmses, label="all data")
        ax.plot(n_contexts, rmses_context, label="context only")
        ax.legend()
        ax.grid()
        ax_ct += 1

    fig.tight_layout()

    return fig
