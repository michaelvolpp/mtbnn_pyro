import os

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from metalearning_benchmarks import Affine1D, Quadratic1D, Sinusoid
from mlp.mlp import MultiLayerPerceptron
from mlp.plotting import plot_metrics, plot_predictions
from mtutils.mtutils import (BM_DICT, collate_data, norm_area_under_curve,
                             split_tasks)


def run_experiment(
    config,
    wandb_run,
):
    ## define metrics for wandb logging
    wandb_run.define_metric(name="adapt/epoch")
    wandb_run.define_metric(name="adapt/*", step_metric="adapt/epoch")
    wandb_run.define_metric(name="eval/n_context")
    wandb_run.define_metric(name="eval/*", step_metric="eval/n_context")

    ## seeding
    torch.manual_seed(config["seed"])

    ## create benchmark
    bm = BM_DICT[config["bm"]](
        n_task=config["n_tasks"],
        n_datapoints_per_task=config["n_points_per_task"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset"],
        seed_x=config["seed_offset"] + 1,
        seed_noise=config["seed_offset"] + 2,
    )
    x, y = collate_data(bm=bm)
    x_pred = np.linspace(
        bm.x_bounds[0, 0] - 0.1 * (bm.x_bounds[0, 1] - bm.x_bounds[0, 0]),
        bm.x_bounds[0, 1] + 0.1 * (bm.x_bounds[0, 1] - bm.x_bounds[0, 0]),
        config["n_points_pred"],
    )[None, :, None].repeat(config["n_tasks"], axis=0)

    ## create model
    mlp = MultiLayerPerceptron(
        d_x=bm.d_x,
        d_y=bm.d_y,
        n_hidden=config["n_hidden"],
        d_hidden=config["d_hidden"],
    )

    ## do inference
    mses = np.zeros(len(config["n_contexts_pred"]))
    mses_context = np.zeros(len(config["n_contexts_pred"]))
    y_preds = []
    learning_curves = []
    for i, n_context in enumerate(config["n_contexts_pred"]):
        print(f"Adapting to tasks (n_context = {n_context:3d})...")
        x_context, y_context, x_target, y_target = split_tasks(
            x=x, y=y, n_context=n_context
        )
        learning_curves.append(
            mlp.adapt(
                x=x_context,
                y=y_context,
                n_epochs=config["n_epochs"],
                initial_lr=config["initial_lr"],
                final_lr=config["final_lr"],
                wandb_run=wandb_run,
            )
        )
        y_preds.append(mlp.predict(x=x_pred))
        mses[i] = mlp.mse(
            x=x_target,
            y=y_target,
        )
        mses_context[i] = mlp.mse(
            x=x_context,
            y=y_context,
        )
        wandb_run.log(
            {
                "eval/n_context": n_context,
                "eval/marg_ll_target": mses[i],
                "eval/marg_ll_context": mses_context[i],
            }
        )
    wandb_run.summary["eval/mse_target_auc"] = norm_area_under_curve(
        x=config["n_contexts_pred"], y=mses
    )
    wandb_run.summary["eval/mse_context_auc"] = norm_area_under_curve(
        x=config["n_contexts_pred"], y=mses_context
    )

    # plot predictions
    if config["plot"]:
        fig = plot_metrics(
            learning_curves=learning_curves,
            mses=mses,
            mses_context=mses_context,
            n_contexts=config["n_contexts_pred"],
        )
        fig = plot_predictions(
            x=x,
            y=y,
            x_pred=x_pred,
            y_preds=y_preds,
            n_contexts=config["n_contexts_pred"],
            max_tasks=config["max_tasks_plot"],
            n_contexts_plot=config["n_contexts_plot"],
        )
        # wandb_run.log({"predictions_plotly": fig})
        wandb_run.log({"predictions_png": wandb.Image(fig)})

        if wandb_run.mode == "disabled":
            plt.show()


def main():
    ## config
    wandb_mode = os.getenv("WANDB_MODE", "disabled")
    smoke_test = os.getenv("SMOKE_TEST", "False") == "True"
    print(f"wandb_mode={wandb_mode}")
    print(f"smoke_test={smoke_test}")
    config = dict(
        model="MLP",
        seed=123,
        # benchmarks
        bm="Sinusoid1D",
        noise_stddev=0.01,
        n_tasks=1,
        n_points_per_task=128,
        seed_offset=1234,
        # model
        n_hidden=1,
        d_hidden=8,
        # training
        n_epochs=5000 if not smoke_test else 100,
        initial_lr=0.1,
        final_lr=0.00001,
        alpha_reg=0.0,
        n_points_pred=100,
        # evaluation
        n_contexts_pred=(
            [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 128]
            if not smoke_test
            else [0, 5, 16]
        ),
        # plot
        plot=True,
        max_tasks_plot=4,
        n_contexts_plot=[0, 5, 10, 25, 50, 100],
    )

    if wandb_mode != "disabled":
        wandb.login()
    with wandb.init(project="mlp_v0", mode=wandb_mode, config=config) as wandb_run:
        config = wandb_run.config
        run_experiment(config=config, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
