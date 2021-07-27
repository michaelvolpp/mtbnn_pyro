import os

import numpy as np
import pyro
import torch
from matplotlib import pyplot as plt
from pyro import poutine
from pyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal
from tqdm import tqdm

from bnn import BNN


def generate_data():
    a, b, c, sigma = 2.0, -1.0, 0.5, 0.1
    x = torch.linspace(0.0, 1.0, 100).reshape(100, 1)
    y = a * x ** 2 + b * x + c + sigma * torch.randn(size=x.shape)
    y = y.squeeze()
    assert y.ndim == 1

    return x, y


def plot_data(x, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle("Data")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.plot(x, y, ".", label="Data", color="b")


def plot_predictions(x, y, pred_summary, type=""):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(f"Data and Predictions ({type})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plot_data(x=x, y=y, ax=ax)
    # TODO: also plot mean = pred_summary["_RETURN"]
    # plt.plot(x, pred_summary["_RETURN"]["mean"], label="Mean Prediction", color="r")
    # plt.fill_between(
    #     x.squeeze(),
    #     pred_summary["_RETURN"]["5%"],
    #     pred_summary["_RETURN"]["95%"],
    #     alpha=0.3,
    #     label="Mean 90% CI",
    #     color="r",
    # )
    plt.plot(x, pred_summary["obs"]["mean"], label="Mean Observation", color="g")
    plt.fill_between(
        x.squeeze(),
        pred_summary["obs"]["5%"],
        pred_summary["obs"]["95%"],
        alpha=0.3,
        label="Observation 90% CI",
        color="g",
    )
    ax.legend()


def train_svi(model, guide, x, y, n_iters, lr):
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model=model, guide=guide, optim=adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    pbar = tqdm(total=n_iters, desc="SVI training")
    for i in range(n_iters):
        loss = svi.step(x=x, y=y)
        if i % 50 == 0:
            pbar.update(50)
            pbar.set_postfix({"ELBO": f"{loss:.4f}"})
    pbar.close()


def predictions_svi(model, guide, x, y, n_svi_iters, n_samples, lr):
    train_svi(model=model, guide=guide, x=x, y=y, n_iters=n_svi_iters, lr=lr)
    predictive = Predictive(model=model, guide=guide, num_samples=n_samples)
    samples = predictive(x)
    # TODO: why do we need this?
    samples = {k: v.squeeze(1) for k, v in samples.items()}
    pred_summary = summarize_samples(samples)
    return pred_summary, samples


def predictions_mcmc(model, x, y, n_samples, n_burn_in_samples):
    mcmc = MCMC(
        kernel=NUTS(model), num_samples=n_samples, warmup_steps=n_burn_in_samples
    )
    mcmc.run(x=x, y=y)
    samples = mcmc.get_samples()
    predictive = Predictive(model=model, posterior_samples=samples)
    samples = {**samples, **predictive(x)}
    pred_summary = summarize_samples(samples)
    return pred_summary, samples


def summarize_samples(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


def make_log_likelihood(model):
    def _log_likelihood(cond_data, *args, **kwargs):
        # returns sum_i log p(obs_i | cond_data)
        conditioned_model = poutine.condition(model, data=cond_data)
        trace = poutine.trace(conditioned_model).get_trace(*args, **kwargs)
        obs_node = trace.nodes["obs"]
        return obs_node["fn"].log_prob(obs_node["value"]).sum()

    return _log_likelihood


def make_log_marginal_likelihood(model):
    def _log_marginal_likelihood(cond_data, *args, **kwargs):
        # returns log p(obs_{1:N})
        log_likelihood_fn = make_log_likelihood(model)
        S = len(cond_data["sigma"])
        log_likelihoods = torch.zeros(S)
        # TODO: vectorize this!
        for s in range(S):
            cur_cond_data = {k: v[s] for k, v in cond_data.items() if k != "obs"}
            log_likelihoods[s] = log_likelihood_fn(cur_cond_data, *args, **kwargs)
        return torch.logsumexp(log_likelihoods, dim=0) - np.log(S)

    return _log_marginal_likelihood


if __name__ == "__main__":
    # seed pyro/torch
    pyro.set_rng_seed(1234)

    # logpath
    logpath = os.path.join(".", "log")
    os.makedirs(logpath, exist_ok=True)

    # data
    x, y = generate_data()
    # plot_data(x, y)
    # plt.show()

    # bnn
    smoke_test = False
    n_iters = 3000 if not smoke_test else 100
    n_samples = 1000 if not smoke_test else 10
    n_burn_in_samples = 200 if not smoke_test else 10
    bnn = BNN(d_in=1, d_out=1, n_hidden=1, d_hidden=5)

    # guides for svi
    guide_map = AutoDelta(bnn)
    guide_svi_diag = AutoDiagonalNormal(bnn)
    guide_svi_mvn = AutoMultivariateNormal(bnn)

    # sample posterior
    # pred_map, samples_map = predictions_svi(
    #     model=bnn,
    #     guide=guide_map,
    #     x=x,
    #     y=y,
    #     n_svi_iters=n_iters,
    #     n_samples=n_samples,
    #     lr=0.01,
    # )
    # for name, value in pyro.get_param_store().items():
    #     print(name, pyro.param(name))
    # pred_svi_diag, samples_svi_diag = predictions_svi(
    #     model=bnn,
    #     guide=guide_svi_diag,
    #     x=x,
    #     y=y,
    #     n_svi_iters=n_iters,
    #     n_samples=n_samples,
    #     lr=0.01,
    # )
    # for name, value in pyro.get_param_store().items():
    #     print(name, pyro.param(name))
    pred_svi_mvn, samples_svi_mvn = predictions_svi(
        model=bnn,
        guide=guide_svi_mvn,
        x=x,
        y=y,
        n_svi_iters=n_iters,
        n_samples=n_samples,
        lr=0.01,
    )
    # for name, value in pyro.get_param_store().items():
    # print(name, pyro.param(name))
    # pred_mcmc, samples_mcmc = predictions_mcmc(
    #     model=bnn, x=x, y=y, n_samples=n_samples, n_burn_in_samples=n_burn_in_samples
    # )

    # store guides
    with open(os.path.join(logpath, "guide_svi_mvn2"), "wb") as f:
        torch.save(guide_svi_mvn.state_dict(), f)

    # print log marginal likelihoods
    # TODO: computation of log-marginal likelihood is inefficient
    # TODO: merge computation of predicitions with computation of log-marginal likelihood
    lml_fn = make_log_marginal_likelihood(model=bnn)
    # lml_map = lml_fn(cond_data=samples_map, x=x, y=y)
    # lml_svi_diag = lml_fn(cond_data=samples_svi_diag, x=x, y=y)
    lml_svi_mvn = lml_fn(cond_data=samples_svi_mvn, x=x, y=y)
    # lml_svi_mcmc = lml_fn(cond_data=samples_mcmc, x=x, y=y)
    # print(f"Marginal log-likelihood (MAP)  = {lml_map:.4f}")
    # print(f"Marginal log-likelihood (Diag) = {lml_svi_diag:.4f}")
    print(f"Marginal log-likelihood (MVN)  = {lml_svi_mvn:.4f}")
    # print(f"Marginal log-likelihood (MCMC) = {lml_svi_mcmc:.4f}")

    # display predictive distribution
    # plot_predictions(x, y, pred_summary=pred_map, type="MAP")
    # plot_predictions(x, y, pred_summary=pred_svi_diag, type="SVI (Diagonal Guide)")
    plot_predictions(x, y, pred_summary=pred_svi_mvn, type="SVI (MVN Guide)")
    # plot_predictions(x, y, pred_summary=pred_mcmc, type="MCMC")
    plt.show()
