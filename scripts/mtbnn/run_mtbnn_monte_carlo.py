"""
Demo script to showcase the functionality of the multi-task Bayesian neural network 
implementation.
"""

import os
import warnings
from sys import int_info

import numpy as np
import pyro
import torch
import wandb
from matplotlib import pyplot as plt
from metalearning_benchmarks.benchmarks.util import normalize_benchmark
from mtutils.mtutils import BM_DICT, collate_data, norm_area_under_curve
from mtutils.mtutils import print_headline_string as prinths
from mtutils.mtutils import print_pyro_parameters, split_tasks, summarize_samples
from wandb.sdk.wandb_init import init

from mtbnn.mtbnn import MultiTaskBayesianNeuralNetwork, _train_prior_monte_carlo
from mtbnn.plotting import plot_distributions, plot_metrics, plot_predictions


def run_experiment(
    config,
    wandb_run,
):
    ## define metrics for wandb logging
    wandb_run.define_metric(name="meta_train/epoch")
    wandb_run.define_metric(name="meta_train/*", step_metric="meta_train/epoch")

    ## seeding
    pyro.set_rng_seed(config["seed_pyro"])

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
    if config["prior_type"] == "fixed":
        do_meta_training = False
        prior_type = "factorized_normal"
    else:
        do_meta_training = True
        prior_type = config["prior_type"]
    mtbnn = MultiTaskBayesianNeuralNetwork(
        d_x=bm_meta.d_x,
        d_y=bm_meta.d_y,
        n_hidden=config["n_hidden"],
        d_hidden=config["d_hidden"],
        noise_stddev=None if config["infer_noise_stddev"] else config["noise_stddev"],
        prior_type=prior_type,
        prior_init=config["prior_init"],
        posterior_init="set_to_prior",  # not relevant for this experiment
    )

    ## obtain predictions on meta data before meta training
    samples_prior_meta_untrained = mtbnn.predict(
        x=x_pred_meta, n_samples=config["n_samples_pred"], guide=None
    )
    pred_summary_prior_meta_untrained = summarize_samples(
        samples=samples_prior_meta_untrained
    )

    ## compute marginal likelihood before training
    marg_ll_meta_untrained = mtbnn.marginal_log_likelihood(
        x=x_meta, y=y_meta, n_samples=config["n_samples_pred"], guide=None
    )
    marg_ll_test_untrained = mtbnn.marginal_log_likelihood(
        x=x_test, y=y_test, n_samples=config["n_samples_pred"], guide=None
    )

    ## print prior parameters
    prinths("Pyro Parameters (before meta training)")
    print_pyro_parameters()

    ## meta training
    prinths("Performing Meta Training...")
    if do_meta_training:
        mtbnn.train()
        mtbnn.unfreeze_prior()
        learning_curve_meta = _train_prior_monte_carlo(
            model=mtbnn,
            x=torch.tensor(x_meta, dtype=torch.float32),
            y=torch.tensor(y_meta, dtype=torch.float32),
            n_epochs=config["n_epochs"],
            n_samples=config["n_samples_pred"],
            initial_lr=config["initial_lr"],
            final_lr=config["final_lr"],
            wandb_run=wandb_run,
            log_identifier="meta_train_monte_carlo",
        )
        mtbnn.freeze_prior()
        mtbnn.eval()
    else:
        print("No meta training performed!")
        learning_curve_meta = None

    ## save model
    # with open("model.onnx", "wb") as f:
    #     mtbnn.export_onnx(f=f)
    # wandb_run.save("model.onnx")

    ## print learned parameters
    prinths("Pyro Parameters (after meta training)")
    print_pyro_parameters()

    ## obtain predictions on meta data after training
    # obtain prior predictions
    samples_prior_meta_trained = mtbnn.predict(
        x=x_pred_meta, n_samples=config["n_samples_pred"], guide=None
    )
    pred_summary_prior_meta_trained = summarize_samples(
        samples=samples_prior_meta_trained
    )

    ## compute marginal likelihood after training
    marg_ll_meta_trained = mtbnn.marginal_log_likelihood(
        x=x_meta, y=y_meta, n_samples=config["n_samples_pred"], guide=None
    )
    marg_ll_test_trained = mtbnn.marginal_log_likelihood(
        x=x_test, y=y_test, n_samples=config["n_samples_pred"], guide=None
    )

    ## print freezed parameters
    prinths("Freezed Pyro Parameters (before adaptation)")
    print_pyro_parameters()

    ## print and log marginal likelihoods
    print(f"{marg_ll_meta_untrained = :.4f}")
    print(f"{marg_ll_meta_trained   = :.4f}")
    print(f"{marg_ll_test_untrained = :.4f}")
    print(f"{marg_ll_test_trained   = :.4f}")
    wandb_run.summary["marg_ll_meta_untrained"] = marg_ll_meta_untrained
    wandb_run.summary["marg_ll_meta_trained"] = marg_ll_meta_trained
    wandb_run.summary["marg_ll_test_untrained"] = marg_ll_test_untrained
    wandb_run.summary["marg_ll_test_trained"] = marg_ll_test_trained

    # plot predictions
    if config["plot"]:
        fig = plot_metrics(
            learning_curve_meta=learning_curve_meta,
            learning_curves_test=None,
            lls=None,
            lls_context=None,
            n_contexts=None,
        )
        fig = plot_predictions(
            x_meta=x_meta,
            y_meta=y_meta,
            x_pred_meta=x_pred_meta,
            # pred_summary_prior_meta_untrained=pred_summary_prior_meta_untrained,
            pred_summary_prior_meta_untrained=None,
            samples_prior_meta_untrained=samples_prior_meta_untrained,
            # pred_summary_prior_meta_trained=pred_summary_prior_meta_trained,
            pred_summary_prior_meta_trained=None,
            samples_prior_meta_trained=samples_prior_meta_trained,
            pred_summary_posterior_meta=None,
            x_test=None,
            y_test=None,
            x_pred_test=None,
            n_contexts_test=None,
            pred_summaries_posterior_test=None,
            n_contexts_plot=None,
            max_tasks=config["max_tasks_plot"],
        )
        wandb_run.log({"predictions_png": wandb.Image(fig)})

        if wandb_run.mode == "disabled":
            plt.show()


def main():
    ## config
    wandb_mode = os.getenv("WANDB_MODE", "online")
    smoke_test = os.getenv("SMOKE_TEST", "True") == "True"
    print(f"wandb_mode={wandb_mode}")
    print(f"smoke_test={smoke_test}")
    config = dict(
        model="MTBNN",
        seed_pyro=123,
        # benchmarks
        bm="Quadratic1D",
        noise_stddev=0.01,
        n_tasks_meta=8,
        n_points_per_task_meta=16,
        n_tasks_test=128,
        n_points_per_task_test=128,
        seed_offset_train=1234,
        seed_offset_test=1235,
        normalize_bm=True,
        # model
        n_hidden=1,
        d_hidden=8,
        infer_noise_stddev=True,
        prior_type="factorized_normal",
        prior_init="as_pytorch_linear",
        # training
        n_epochs=5000 if not smoke_test else 100,
        initial_lr=0.1,
        final_lr=0.00001,
        n_points_pred=100,
        n_samples_pred=1000 if not smoke_test else 100,
        # plot
        plot=True,
        max_tasks_plot=np.inf,
    )

    if wandb_mode != "disabled":
        wandb.login()
    with wandb.init(project="mtbnn_mc", mode=wandb_mode, config=config) as wandb_run:
        config = wandb_run.config
        run_experiment(config=config, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
