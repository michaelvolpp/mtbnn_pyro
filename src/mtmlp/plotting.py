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
    y_pred: np.ndarray,
    ax: None,
    max_tasks: int,
    n_context: Optional[np.ndarray] = None,
):
    # assert that all inputs have the same number of dimensions
    assert x.ndim == y.ndim == x_pred.ndim == y_pred.ndim
    # check that all inputs are one-dimensional
    assert x.shape[-1] == y.shape[-1] == x_pred.shape[-1] == y_pred.shape[-1] == 1
    # check that all inputs have the same number of tasks
    assert x.shape[0] == y.shape[0] == y_pred.shape[0]
    # squeeze data dimension
    (x, y, x_pred, y_pred) = (
        x.squeeze(-1),
        y.squeeze(-1),
        x_pred.squeeze(-1),
        y_pred.squeeze(-1),
    )

    ## prepare plot
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    ## plot
    n_task = x.shape[0]
    for l in range(n_task):
        if l == max_tasks:
            break
        line = ax.plot(x_pred[l, :], y_pred[l, :])[0]
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
    ax.grid()


def plot_predictions(
    x,
    y,
    n_contexts,
    x_pred,
    y_preds,
    n_contexts_plot,
    max_tasks,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(n_contexts_plot),
        figsize=(16, 8),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(f"Prior and Posterior Predictions")

    plt_ct = -1
    for i, n_context in enumerate(n_contexts):
        if n_context not in n_contexts_plot:
            continue

        plt_ct += 1
        ax = axes[0, plt_ct]
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Prediction\n(test data, n_ctx={n_context:d})")
        plot_predictions_for_one_set_of_tasks(
            x=x,
            y=y,
            x_pred=x_pred,
            y_pred=y_preds[i],
            n_context=n_context,
            ax=ax,
            max_tasks=max_tasks,
        )
    fig.tight_layout()

    return fig


def plot_metrics(learning_curves, mses, mses_context, n_contexts):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), squeeze=False)
    fig.suptitle("Metrics")

    ax = axes[0, 0]
    ax.set_title("Learning Curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log") 
    for learning_curve, n_context in zip(learning_curves, n_contexts):
        ax.plot(
            np.arange(len(learning_curve)),
            learning_curve,
            label=f"n_ctx={n_context:d}",
        )
    ax.legend()
    ax.grid()

    ax = axes[0, 1]
    ax.set_title("MSE")
    ax.set_xlabel("n_context")
    ax.set_xticks(n_contexts)
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.plot(n_contexts, mses, label="all data")
    ax.plot(n_contexts, mses_context, label="context only")
    ax.legend()
    ax.grid()

    fig.tight_layout()

    return fig
