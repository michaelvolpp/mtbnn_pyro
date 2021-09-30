"""
Demo script to showcase the functionality of the multi-task Bayesian neural network 
implementation.
"""

import os
import warnings

import numpy as np
import pyro
from matplotlib import pyplot as plt
from metalearning_benchmarks import Affine1D, Quadratic1D, Sinusoid

import wandb
from mtbnn import MultiTaskBayesianNeuralNetwork
from plotting import plot_distributions, plot_metrics, plot_predictions
from util import collate_data
from util import print_headline_string as prinths
from util import print_pyro_parameters, split_tasks, summarize_samples

BM_DICT = {"Affine1D": Affine1D, "Quadratic1D": Quadratic1D, "Sinusoid1D": Sinusoid}


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
        prior_type = "isotropic_normal"
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
    )

    ## obtain predictions on meta data before meta training
    samples_prior_meta_untrained = mtbnn.predict(
        x=x_pred_meta, n_samples=config["n_samples_pred"], guide="prior"
    )
    pred_summary_prior_meta_untrained = summarize_samples(
        samples=samples_prior_meta_untrained
    )

    ## print prior parameters
    prinths("Pyro Parameters (before meta training)")
    print_pyro_parameters()

    ## meta training
    prinths("Performing Meta Training...")
    if do_meta_training:
        learning_curve_meta = mtbnn.meta_train(
            x=x_meta,
            y=y_meta,
            n_epochs=config["n_epochs"],
            initial_lr=config["initial_lr"],
            final_lr=config["final_lr"],
            alpha_reg=config["alpha_reg"],
            wandb_run=wandb_run,
        )
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
        x=x_pred_meta, n_samples=config["n_samples_pred"], guide="prior"
    )
    pred_summary_prior_meta_trained = summarize_samples(
        samples=samples_prior_meta_trained
    )
    # obtain posterior predictions
    samples_posterior_meta = mtbnn.predict(
        x=x_pred_meta, n_samples=config["n_samples_pred"], guide="meta"
    )
    pred_summary_posterior_meta = summarize_samples(samples=samples_posterior_meta)

    # print freezed parameters
    prinths("Freezed Pyro Parameters (before adaptation)")
    print_pyro_parameters()

    ## do inference on test task
    lls = np.zeros(len(config["n_contexts_pred"]))
    lls_context = np.zeros(len(config["n_contexts_pred"]))
    pred_summaries_posteriors_test, samples_posteriors_test = [], []
    learning_curves_test = []
    for i, n_context in enumerate(config["n_contexts_pred"]):
        prinths(f"Adapting to test tasks (n_context = {n_context:3d})...")
        x_context, y_context, x_target, y_target = split_tasks(
            x=x_test, y=y_test, n_context=n_context
        )
        learning_curves_test.append(
            mtbnn.adapt(
                x=x_context,
                y=y_context,
                n_epochs=config["n_epochs"],
                initial_lr=config["initial_lr"],
                final_lr=config["final_lr"],
                wandb_run=wandb_run,
            )
        )
        lls[i] = mtbnn.marginal_log_likelihood(
            x=x_target,
            y=y_target,
            n_samples=config["n_samples_pred"],
            guide_choice="test",
        )
        lls_context[i] = mtbnn.marginal_log_likelihood(
            x=x_context,
            y=y_context,
            n_samples=config["n_samples_pred"],
            guide_choice="test",
        )
        cur_samples_posterior_test = mtbnn.predict(
            x=x_pred_test, n_samples=config["n_samples_pred"], guide="test"
        )
        cur_pred_summary_posterior_test = summarize_samples(
            samples=cur_samples_posterior_test
        )
        pred_summaries_posteriors_test.append(cur_pred_summary_posterior_test)
        samples_posteriors_test.append(cur_samples_posterior_test)
        wandb.log(
            {
                "eval/n_context": n_context,
                "eval/marg_ll_target": lls[i],
                "eval/marg_ll_context": lls_context[i],
            }
        )

    prinths("Freezed Pyro Parameters (after adaptation)")
    print_pyro_parameters()

    # plot predictions
    if config["plot"]:
        fig = plot_metrics(
            learning_curve_meta=learning_curve_meta,
            learning_curves_test=learning_curves_test,
            lls=lls,
            lls_context=lls_context,
            n_contexts=config["n_contexts_pred"],
        )
        fig = plot_predictions(
            x_meta=x_meta,
            y_meta=y_meta,
            x_pred_meta=x_pred_meta,
            x_test=x_test,
            y_test=y_test,
            x_pred_test=x_pred_test,
            n_contexts_test=config["n_contexts_pred"],
            pred_summary_prior_meta_untrained=pred_summary_prior_meta_untrained,
            pred_summary_prior_meta_trained=pred_summary_prior_meta_trained,
            pred_summary_posterior_meta=pred_summary_posterior_meta,
            pred_summaries_posterior_test=pred_summaries_posteriors_test,
            max_tasks=config["max_tasks_plot"],
            n_contexts_plot=config["n_contexts_plot"],
        )
        # wandb_run.log({"predictions_plotly": fig})
        wandb_run.log({"predictions_png": wandb.Image(fig)})

        if config["n_hidden"] == 0:
            # plot prior and posterior distributions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if isinstance(bm_meta, Affine1D):
                    bm_meta_params = np.zeros(config["n_tasks_meta"])
                    bm_test_params = np.zeros(config["n_tasks_test"])
                    for l, task in enumerate(bm_meta):
                        bm_meta_params[l] = task.param[0]
                    for l, task in enumerate(bm_test):
                        bm_test_params[l] = task.param[0]
                else:
                    bm_meta_params, bm_test_params = None, None
                fig = plot_distributions(
                    site_name="_wb",
                    site_idx=0,
                    bm_meta_params=bm_meta_params,
                    bm_test_params=bm_test_params,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posteriors_test=samples_posteriors_test,
                    n_contexts_test=config["n_contexts_pred"],
                    max_tasks=config["max_tasks_plot"],
                    n_contexts_plot=config["n_contexts_plot"],
                )
                # wandb_run.log({"latent_distribution_w_plotly": fig})
                wandb_run.log({"latent_distribution_w_png": wandb.Image(fig)})

                if isinstance(bm_meta, Affine1D):
                    bm_meta_params = np.zeros(config["n_tasks_meta"])
                    bm_test_params = np.zeros(config["n_tasks_test"])
                    for l, task in enumerate(bm_meta):
                        bm_meta_params[l] = task.param[1]
                    for l, task in enumerate(bm_test):
                        bm_test_params[l] = task.param[1]
                else:
                    bm_meta_params, bm_test_params = None, None
                fig = plot_distributions(
                    site_name="_wb",
                    site_idx=1,
                    bm_meta_params=bm_meta_params,
                    bm_test_params=bm_test_params,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posteriors_test=samples_posteriors_test,
                    n_contexts_test=config["n_contexts_pred"],
                    max_tasks=config["max_tasks_plot"],
                    n_contexts_plot=config["n_contexts_plot"],
                )
                # wandb_run.log({"latent_distribution_b_plotly": fig})
                wandb_run.log({"latent_distribution_b_png": wandb.Image(fig)})

        if wandb_run.mode == "disabled":
            plt.show()


def main():
    ## config
    wandb_mode = os.getenv("WANDB_MODE", "online")
    smoke_test = os.getenv("SMOKE_TEST", "False") == "True"
    print(f"wandb_mode={wandb_mode}")
    print(f"smoke_test={smoke_test}")
    config = dict(
        seed_pyro=123,
        # benchmarks
        bm="Affine1D",
        noise_stddev=0.01,
        n_tasks_meta=8,
        n_points_per_task_meta=16,
        n_tasks_test=128,
        n_points_per_task_test=128,
        seed_offset_train=1234,
        seed_offset_test=1235,
        # model
        n_hidden=1,
        d_hidden=8,
        infer_noise_stddev=True,
        prior_type="fixed",
        # training
        n_epochs=5000 if not smoke_test else 100,
        initial_lr=0.1,
        final_lr=0.00001,
        alpha_reg=0.0,
        n_points_pred=100,
        n_samples_pred=1000 if not smoke_test else 100,
        # evaluation
        n_contexts_pred=(
            [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 128]
            if not smoke_test
            else [0, 5, 16]
        ),
        # plot
        plot=True,
        max_tasks_plot=4,
        n_contexts_plot=[5, 10, 50],
    )

    if wandb_mode != "disabled":
        wandb.login()
    with wandb.init(project="mtbnn_v0", mode=wandb_mode, config=config) as wandb_run:
        config = wandb_run.config
        run_experiment(config=config, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
