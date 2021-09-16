"""
Demo script to showcase the functionality of the multi-task Bayesian neural network 
implementation.
"""

import warnings
import numpy as np
import pyro
from matplotlib import pyplot as plt
from metalearning_benchmarks import Affine1D, Quadratic1D

from mtbnn import MultiTaskBayesianNeuralNetwork
from plotting import plot_distributions, plot_metrics, plot_predictions
from util import collate_data, print_pyro_parameters, split_tasks, summarize_samples
from util import print_headline_string as prinths


def main():
    # TODO: use exact prior/posterior distributions, not the KDE (e.g., prior is task-independent!)
    # TODO: implement more complex priors (e.g., not factorized across layers?)

    ## flags, constants
    pyro.set_rng_seed(123)
    plot = True
    smoke_test = False
    # benchmarks
    bm = Quadratic1D
    noise_stddev = 0.01
    n_tasks_meta = 8
    n_points_per_task_meta = 16
    n_tasks_test = 128
    n_points_per_task_test = 128
    # model
    n_hidden = 1
    d_hidden = 8
    infer_noise_stddev = True
    # prior_type = "multivariate_gaussian"
    prior_type = "diagonal_gaussian"
    # training
    do_meta_training = True
    n_epochs = 5000 if not smoke_test else 100
    initial_lr = 0.1
    final_lr = 0.00001
    alpha_reg = 0.0
    # evaluation
    n_contexts = (
        np.array([0, 1, 2, 5, 10, n_points_per_task_test])
        if not smoke_test
        else np.array([0, 5, 10])
    )
    n_pred = 100
    n_samples = 1000 if not smoke_test else 100
    max_plot_tasks = 5

    ## create benchmarks
    # meta benchmark
    bm_meta = bm(
        n_task=n_tasks_meta,
        n_datapoints_per_task=n_points_per_task_meta,
        output_noise=noise_stddev,
        seed_task=1234,
        seed_x=1235,
        seed_noise=1236,
    )
    x_meta, y_meta = collate_data(bm=bm_meta)
    x_pred_meta = np.linspace(-1.5, 1.5, n_pred)[None, :, None].repeat(
        n_tasks_meta, axis=0
    )
    # test benchmark
    bm_test = bm(
        n_task=n_tasks_test,
        n_datapoints_per_task=n_points_per_task_test,
        output_noise=noise_stddev,
        seed_task=1235,
        seed_x=1236,
        seed_noise=1237,
    )
    x_test, y_test = collate_data(bm=bm_test)
    x_pred_test = np.linspace(-1.5, 1.5, n_pred)[None, :, None].repeat(
        n_tasks_test, axis=0
    )

    ## create model
    mtbnn = MultiTaskBayesianNeuralNetwork(
        d_x=bm_meta.d_x,
        d_y=bm_meta.d_y,
        n_hidden=n_hidden,
        d_hidden=d_hidden,
        noise_stddev=None if infer_noise_stddev else noise_stddev,
        prior_type=prior_type,
    )

    ## obtain predictions on meta data before meta training
    samples_prior_meta_untrained = mtbnn.predict(
        x=x_pred_meta, n_samples=n_samples, guide="prior"
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
            n_epochs=n_epochs,
            initial_lr=initial_lr,
            final_lr=final_lr,
            alpha_reg=alpha_reg,
        )
    else:
        print("No meta training performed!")
        learning_curve_meta = None

    ## print learned parameters
    prinths("Pyro Parameters (after meta training)")
    print_pyro_parameters()

    ## obtain predictions on meta data after training
    # obtain prior predictions
    samples_prior_meta_trained = mtbnn.predict(
        x=x_pred_meta, n_samples=n_samples, guide="prior"
    )
    pred_summary_prior_meta_trained = summarize_samples(
        samples=samples_prior_meta_trained
    )
    # obtain posterior predictions
    samples_posterior_meta = mtbnn.predict(
        x=x_pred_meta, n_samples=n_samples, guide="meta"
    )
    pred_summary_posterior_meta = summarize_samples(samples=samples_posterior_meta)

    # print freezed parameters
    prinths("Freezed Pyro Parameters (before adaptation)")
    print_pyro_parameters()

    ## do inference on test task
    lls = np.zeros(n_contexts.shape)
    lls_context = np.zeros(n_contexts.shape)
    pred_summaries_posteriors_test, samples_posteriors_test = [], []
    learning_curves_test = []
    for i, n_context in enumerate(n_contexts):
        prinths(f"Adapting to test tasks (n_context = {n_context:3d})...")
        x_context, y_context, x_target, y_target = split_tasks(
            x=x_test, y=y_test, n_context=n_context
        )
        learning_curves_test.append(
            mtbnn.adapt(
                x=x_context,
                y=y_context,
                n_epochs=n_epochs,
                initial_lr=initial_lr,
                final_lr=final_lr,
            )
        )
        lls[i] = mtbnn.marginal_log_likelihood(
            x=x_target,
            y=y_target,
            n_samples=n_samples,
            guide_choice="test",
        )
        lls_context[i] = mtbnn.marginal_log_likelihood(
            x=x_context,
            y=y_context,
            n_samples=n_samples,
            guide_choice="test",
        )
        cur_samples_posterior_test = mtbnn.predict(
            x=x_pred_test, n_samples=n_samples, guide="test"
        )
        cur_pred_summary_posterior_test = summarize_samples(
            samples=cur_samples_posterior_test
        )
        pred_summaries_posteriors_test.append(cur_pred_summary_posterior_test)
        samples_posteriors_test.append(cur_samples_posterior_test)

    prinths("Freezed Pyro Parameters (after adaptation)")
    print_pyro_parameters()

    # plot predictions
    if plot:
        plot_metrics(
            learning_curve_meta=learning_curve_meta,
            learning_curves_test=learning_curves_test,
            lls=lls,
            lls_context=lls_context,
            n_contexts=n_contexts,
        )
        plot_predictions(
            x_meta=x_meta,
            y_meta=y_meta,
            x_pred_meta=x_pred_meta,
            x_test=x_test,
            y_test=y_test,
            x_pred_test=x_pred_test,
            n_contexts_test=n_contexts,
            pred_summary_prior_meta_untrained=pred_summary_prior_meta_untrained,
            pred_summary_prior_meta_trained=pred_summary_prior_meta_trained,
            pred_summary_posterior_meta=pred_summary_posterior_meta,
            pred_summaries_posterior_test=pred_summaries_posteriors_test,
            max_tasks=max_plot_tasks,
        )

        if n_hidden == 0:
            # plot prior and posterior distributions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if isinstance(bm_meta, Affine1D):
                    bm_meta_params = np.zeros(n_tasks_meta)
                    bm_test_params = np.zeros(n_tasks_test)
                    for l, task in enumerate(bm_meta):
                        bm_meta_params[l] = task.param[0]
                    for l, task in enumerate(bm_test):
                        bm_test_params[l] = task.param[1]
                else:
                    bm_meta_params, bm_test_params = None, None
                plot_distributions(
                    site_name="net.0.weight",
                    bm_meta_params=bm_meta_params,
                    bm_test_params=bm_test_params,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posteriors_test=samples_posteriors_test,
                    n_contexts_test=n_contexts,
                )

                if isinstance(bm_meta, Affine1D):
                    bm_meta_params = np.zeros(n_tasks_meta)
                    bm_test_params = np.zeros(n_tasks_test)
                    for l, task in enumerate(bm_meta):
                        bm_meta_params[l] = task.param[1]
                    for l, task in enumerate(bm_test):
                        bm_test_params[l] = task.param[1]
                else:
                    bm_meta_params, bm_test_params = None, None
                plot_distributions(
                    site_name="net.0.bias",
                    bm_meta_params=bm_meta_params,
                    bm_test_params=bm_test_params,
                    samples_prior_meta_untrained=samples_prior_meta_untrained,
                    samples_prior_meta_trained=samples_prior_meta_trained,
                    samples_posterior_meta=samples_posterior_meta,
                    samples_posteriors_test=samples_posteriors_test,
                    n_contexts_test=n_contexts,
                )

        plt.show()


if __name__ == "__main__":
    main()
