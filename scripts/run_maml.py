"""
Run the MAML benchmark.
"""

import os

import learn2learn as l2l
import numpy as np
import torch
from torch._C import dtype
import wandb
from matplotlib import pyplot as plt
from metalearning_benchmarks.util import normalize_benchmark
from mlp.mlp import MultiLayerPerceptron
from mtutils.mtutils import BM_DICT, collate_data, norm_area_under_curve
from mtutils.mtutils import print_headline_string as prinths
from mtutils.mtutils import split_tasks

from mtbnn.plotting import plot_metrics
from mtmlp.plotting import plot_predictions

# stuff to investigate
# - Tanh vs Relu
# - n_task -> 160000 different sines vs. 8
# - n_task defines n_epochs
# - n_points_per_task -> 20 vs 16
# - batched training
# - network size
# - do we overfit to the meta training data (we do not validate during training)?
# - random context-target splits (each episode)?


def train_maml(
    model, x, y, n_context, n_epochs, n_adapt_steps, adapt_lr, meta_lr, wandb_run
):
    """
    Train model using MAML on data x, y using n_context context points.
    """
    maml = l2l.algorithms.MAML(
        model,
        lr=adapt_lr,
        first_order=False,
        allow_unused=True,
    )
    optim = torch.optim.Adam(maml.parameters(), meta_lr)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    n_tasks = x.shape[0]

    # divide the data into context and target sets
    x_context, y_context, x_target, y_target = split_tasks(
        x=x,
        y=y,
        n_context=n_context,
    )
    x_context, y_context, x_target, y_target = (
        torch.tensor(x_context, dtype=torch.float32),
        torch.tensor(y_context, dtype=torch.float32),
        torch.tensor(x_target, dtype=torch.float32),
        torch.tensor(y_target, dtype=torch.float32),
    )

    # for each iteration
    learning_curve_meta = []
    for i in range(n_epochs):
        meta_train_loss = 0.0

        # for each task in the batch
        for l in range(n_tasks):
            learner = maml.clone()

            for _ in range(n_adapt_steps):  # adaptation_steps
                context_preds = learner(x_context[l : l + 1])
                context_loss = loss_fn(context_preds, y_context[l : l + 1])
                learner.adapt(context_loss)

            target_preds = learner(x_target[l : l + 1])
            target_loss = loss_fn(target_preds, y_target[l : l + 1])
            meta_train_loss += target_loss

        meta_train_loss = meta_train_loss / n_tasks
        learning_curve_meta.append(meta_train_loss.item())

        # log
        wandb_run.log(
            {
                f"meta_train/epoch": i,
                f"meta_train/loss_n_context_{n_context:03d}": meta_train_loss.item(),
            }
        )
        if i % 100 == 0 or i == n_epochs - 1:
            print(f"Epoch = {i:04d} | Meta Train Loss = {meta_train_loss.item():.4f}")

        optim.zero_grad()
        meta_train_loss.backward()
        optim.step()

    return learning_curve_meta


def evaluate_maml(model, x, y, n_context, x_pred, n_adapt_steps, adapt_lr):
    """
    Evaluate RMSE of model on data x, y using n_context context_points.
    Also return predictions of adapted model for each task.
    """
    maml = l2l.algorithms.MAML(
        model,
        lr=adapt_lr,
        first_order=False,
        allow_unused=True,
    )
    loss_fn = torch.nn.MSELoss(reduction="mean")
    n_tasks = x.shape[0]

    # divide the data into context and target sets
    x_context, y_context, x_target, y_target = split_tasks(
        x=x,
        y=y,
        n_context=n_context,
    )
    x_context, y_context, x_target, y_target = (
        torch.tensor(x_context, dtype=torch.float32),
        torch.tensor(y_context, dtype=torch.float32),
        torch.tensor(x_target, dtype=torch.float32),
        torch.tensor(y_target, dtype=torch.float32),
    )

    mse = 0.0
    mse_context = 0.0
    cur_y_pred = []
    for l in range(n_tasks):
        learner = maml.clone()

        # adapt
        for _ in range(n_adapt_steps):  # adaptation_steps
            context_preds = learner(x_context[l : l + 1])
            context_loss = loss_fn(context_preds, y_context[l : l + 1])
            learner.adapt(context_loss)

        # predict
        cur_y_pred.append(learner.predict(x=x_pred[l : l + 1]))

        # mse
        mse += learner.mse(x=x_target[l : l + 1].numpy(), y=y_target[l : l + 1].numpy())
        mse_context += learner.mse(
            x=x_context[l : l + 1].numpy(), y=y_context[l : l + 1].numpy()
        )

    y_pred = np.concatenate(cur_y_pred, axis=0)
    rmse = np.sqrt(mse / n_tasks)
    rmse_context = np.sqrt(mse_context / n_tasks)

    return rmse, rmse_context, y_pred


def run_experiment(
    config,
    wandb_run,
):
    ## define metrics for wandb logging
    wandb_run.define_metric(name="meta_train/epoch")
    wandb_run.define_metric(name="meta_train/*", step_metric="meta_train/epoch")
    wandb_run.define_metric(name="adapt/epoch")
    wandb_run.define_metric(name="adapt/*", step_metric="adapt/epoch")
    wandb_run.define_metric(name="eval/n_context")
    wandb_run.define_metric(name="eval/*", step_metric="eval/n_context")

    ## seeding
    torch.manual_seed(config["seed"])

    ## create benchmarks
    # meta benchmark
    bm_meta = BM_DICT[config["bm"]](
        n_task=config["n_tasks_meta"],
        n_datapoints_per_task=config["n_points_per_task_meta"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset_train"],
        seed_x=config["seed_offset_train"] + 1,
        seed_noise=config["seed_offset_train"] + 2,
    )
    if config["normalize_bm"]:
        bm_meta = normalize_benchmark(benchmark=bm_meta)
    x_meta, y_meta = collate_data(bm=bm_meta)
    x_pred_meta = np.linspace(
        bm_meta.x_bounds[0, 0]
        - 0.1 * (bm_meta.x_bounds[0, 1] - bm_meta.x_bounds[0, 0]),
        bm_meta.x_bounds[0, 1]
        + 0.1 * (bm_meta.x_bounds[0, 1] - bm_meta.x_bounds[0, 0]),
        config["n_points_pred"],
    )[None, :, None].repeat(config["n_tasks_meta"], axis=0)
    # test benchmark
    bm_test = BM_DICT[config["bm"]](
        n_task=config["n_tasks_test"],
        n_datapoints_per_task=config["n_points_per_task_test"],
        output_noise=config["noise_stddev"],
        seed_task=config["seed_offset_test"],
        seed_x=config["seed_offset_test"] + 1,
        seed_noise=config["seed_offset_test"] + 2,
    )
    if config["normalize_bm"]:
        bm_test = normalize_benchmark(benchmark=bm_test)
    x_test, y_test = collate_data(bm=bm_test)
    x_pred_test = np.linspace(
        bm_test.x_bounds[0, 0]
        - 0.1 * (bm_test.x_bounds[0, 1] - bm_test.x_bounds[0, 0]),
        bm_test.x_bounds[0, 1]
        + 0.1 * (bm_test.x_bounds[0, 1] - bm_test.x_bounds[0, 0]),
        config["n_points_pred"],
    )[None, :, None].repeat(config["n_tasks_test"], axis=0)

    ## create model
    model = MultiLayerPerceptron(
        d_x=bm_meta.d_x,
        d_y=bm_meta.d_y,
        hidden_units=config["hidden_units"],
        f_act=config["f_act"],
    )

    ## meta training
    prinths("Performing Meta Training...")
    learning_curve_meta = train_maml(
        model=model,
        x=x_meta,
        y=y_meta,
        n_context=config["n_context_meta"],
        n_epochs=config["n_epochs"],
        n_adapt_steps=config["n_adapt_steps"],
        adapt_lr=config["adapt_lr"],
        meta_lr=config["meta_lr"],
        wandb_run=wandb_run,
    )

    ## evaluate on meta tasks (for n_context = n_context_meta)
    rmse_meta, rmse_context_meta, y_pred_meta = evaluate_maml(
        model=model,
        x=x_meta,
        y=y_meta,
        x_pred=x_pred_meta,
        n_context=config["n_context_meta"],
        n_adapt_steps=config["n_adapt_steps"],
        adapt_lr=config["adapt_lr"],
    )
    y_preds_meta = y_pred_meta[None, :, :]

    ## evaluate on test tasks (for varying n_context)
    rmses_test = np.zeros(len(config["n_contexts_pred"]))
    rmses_context_test = np.zeros(len(config["n_contexts_pred"]))
    y_preds_test = []
    for i, n_context in enumerate(config["n_contexts_pred"]):
        print(f"Adapting to tasks (n_context = {n_context:3d})...")

        cur_rmse_test, cur_rmse_context_test, cur_y_pred_test = evaluate_maml(
            model=model,
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            n_context=n_context,
            n_adapt_steps=config["n_adapt_steps"],
            adapt_lr=config["adapt_lr"],
        )

        # log
        rmses_test[i] = cur_rmse_test
        rmses_context_test[i] = cur_rmse_context_test
        y_preds_test.append(cur_y_pred_test)
        wandb_run.log(
            {
                "eval/n_context": n_context,
                "eval/rmse": rmses_test[i],
                "eval/rmse_context": rmses_context_test[i],
            }
        )
    y_preds_test = np.stack(y_preds_test, axis=0)

    ## log summaries
    wandb_run.summary["meta_train/rmse_meta_n_ctx_meta"] = rmse_meta
    wandb_run.summary["meta_train/rmse_context_meta_n_ctx_meta"] = rmse_context_meta
    wandb_run.summary["eval/rmse_test_n_ctx_meta"] = rmses_test[
        config["n_contexts_pred"].index(config["n_context_meta"])
    ]
    wandb_run.summary["eval/rmse_context_test_n_ctx_meta"] = rmses_context_test[
        config["n_contexts_pred"].index(config["n_context_meta"])
    ]
    wandb_run.summary["eval/rmse_test_auc"] = norm_area_under_curve(
        x=config["n_contexts_pred"], y=rmses_test
    )
    wandb_run.summary["eval/rmse_context_test_auc"] = norm_area_under_curve(
        x=config["n_contexts_pred"][1:], y=rmses_context_test[1:]
    )

    ## plot predictions
    if config["plot"]:
        fig = plot_metrics(
            learning_curve_meta=learning_curve_meta,
            learning_curves_test=None,
            rmses=rmses_test,
            rmses_context=rmses_context_test,
            n_contexts=config["n_contexts_pred"],
        )
        fig = plot_predictions(
            x=x_meta,
            y=y_meta,
            x_pred=x_pred_meta,
            y_preds=y_preds_meta,
            n_contexts=[config["n_context_meta"]],
            max_tasks=config["max_tasks_plot"],
            n_contexts_plot=[config["n_context_meta"]],
            dataset_name="meta",
        )
        wandb_run.log({"predictions_meta_png": wandb.Image(fig)})
        fig = plot_predictions(
            x=x_test,
            y=y_test,
            x_pred=x_pred_test,
            y_preds=y_preds_test,
            n_contexts=config["n_contexts_pred"],
            max_tasks=config["max_tasks_plot"],
            n_contexts_plot=config["n_contexts_plot"],
            dataset_name="test",
        )
        wandb_run.log({"predictions_test_png": wandb.Image(fig)})

        if wandb_run.mode == "disabled":
            plt.show()


def main():
    ## config
    wandb_mode = os.getenv("WANDB_MODE", "disabled")
    smoke_test = os.getenv("SMOKE_TEST", "False") == "True"
    print(f"wandb_mode={wandb_mode}")
    print(f"smoke_test={smoke_test}")
    config = dict(
        model="MAML",
        seed=123,
        # benchmarks
        bm="Affine1D",
        noise_stddev=0.01,
        n_tasks_meta=8,
        n_points_per_task_meta=16,
        n_tasks_test=128,
        n_points_per_task_test=128,
        seed_offset_train=1234,
        seed_offset_test=1235,
        normalize_bm=True,
        # model
        hidden_units=[8],
        f_act="relu",
        # training
        n_epochs=10000 if not smoke_test else 100,
        adapt_lr=0.01,
        meta_lr=0.001,
        n_adapt_steps=5,
        n_context_meta=8,
        # evaluation
        n_points_pred=100,
        n_contexts_pred=(
            [0, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 128]
            if not smoke_test
            else [0, 5, 8, 10, 50, 128]
        ),
        # plot
        plot=True,
        max_tasks_plot=4,
        n_contexts_plot=[0, 5, 8, 10, 50],
    )

    if wandb_mode != "disabled":
        wandb.login()
    with wandb.init(project="mtbnn_v0", mode=wandb_mode, config=config) as wandb_run:
        config = wandb_run.config
        run_experiment(config=config, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
