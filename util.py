"""
Utility functions for (multi-task) Bayesian neural networks.
"""
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
import numpy as np
import pyro
from scipy import integrate


def collate_data(bm: MetaLearningBenchmark):
    x = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_x))
    y = np.zeros((bm.n_task, bm.n_points_per_task, bm.d_y))
    for l, task in enumerate(bm):
        x[l, :] = task.x
        y[l, :] = task.y
    return x, y


def summarize_samples(samples: dict) -> dict:
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": np.mean(v, axis=0),
            "std": np.std(v, axis=0),
            "5%": np.percentile(v, 5, axis=0),
            "95%": np.percentile(v, 95, axis=0),
        }
    return site_stats


def print_pyro_parameters() -> None:
    first = True
    for name in pyro.get_param_store().keys():
        if first:
            first = False
        else:
            print("\n")
        print(
            f"name  = {name}"
            f"\nshape = {pyro.param(name).shape}"
            f"\nvalue = {pyro.param(name)}"
        )


def split_tasks(x, y, n_context):
    x_context, y_context = x[:, :n_context, :], y[:, :n_context, :]
    # TODO: use all data as target?
    x_target, y_target = x, y

    return x_context, y_context, x_target, y_target


def print_headline_string(string):
    string = "*** " + string + " ***"
    starline = "*" * len(string)

    print("\n")
    print(starline)
    print(string)
    print(starline)


def norm_area_under_curve(x, y):
    """
    Computes the area under the curve f(x) = y using the trapeziodal rule,
    normalized by the length of the domain.
    """
    assert len(np.unique(x)) == len(x), "x contains duplicate values!"
    domain_length = max(x) - min(x)
    if domain_length == 0.0:
        return None

    area = integrate.trapezoid(x=x, y=y)
    normalized_area = area / domain_length

    return normalized_area


if __name__ == "__main__":
    x = [1.0, 2.0]
    y = [-100.0, 100.1]
    print(norm_area_under_curve(x=x, y=y))
