import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.nn import PyroModule, PyroParam
from torch.distributions import constraints
from matplotlib import pyplot as plt


def check_consistency(L, N, y):
    assert not ((N is None) and (L is not None))
    assert not ((L is None) and (N is not None))
    assert not ((N is None) and (y is None))
    if N is None:
        L = y.shape[0]
        N = y.shape[1]
    if y is not None:
        assert y.ndim == 3
        assert y.shape[0] == L
        assert y.shape[1] == N

    return L, N


class GaussianPrior(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init):
        super().__init__()

        self.mu_z = PyroParam(
            init_value=torch.tensor([mu_z_init]),
            constraint=constraints.real,
        )
        self.sigma_z = PyroParam(
            init_value=torch.tensor([sigma_z_init]),
            constraint=constraints.positive,
        )

    def forward(self):
        prior = dist.Normal(loc=self.mu_z, scale=self.sigma_z).to_event(1)
        z = pyro.sample("z", fn=prior)
        return z


class GaussianLikelihood(PyroModule):
    def __init__(self, sigma_n):
        super().__init__()

        self.sigma_n = sigma_n

    def forward(self, z, y=None):
        likelihood = dist.Normal(loc=z, scale=self.sigma_n).to_event(1)
        obs = pyro.sample("obs", fn=likelihood, obs=y)
        return obs


class GlobalLVMGaussianPriorGuide(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init):
        super().__init__()
        self.prior = GaussianPrior(mu_z_init=mu_z_init, sigma_z_init=sigma_z_init)

    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)

        with pyro.plate("tasks", size=L, dim=-2):
            self.prior()


class GlobalLVM(PyroModule):
    def __init__(self, mu_z_init, sigma_z_init, sigma_n):
        super().__init__()
        self.prior = GaussianPrior(mu_z_init=mu_z_init, sigma_z_init=sigma_z_init)
        self.likelihood = GaussianLikelihood(sigma_n=sigma_n)

    def forward(self, L=None, N=None, y=None):
        L, N = check_consistency(L=L, N=N, y=y)

        with pyro.plate("tasks", size=L, dim=-2):
            z = self.prior()
            with pyro.plate("data", size=N, dim=-1):
                obs = self.likelihood(z=z, y=y)

        return obs


def compute_log_iw(y, S, model, guide):
    assert y.ndim == 3
    L = y.shape[0]
    N = y.shape[1]

    guide = pyro.plate("batch", size=S, dim=-3)(guide)
    model = pyro.plate("batch", size=S, dim=-3)(model)
    guide_trace = poutine.trace(guide).get_trace(L=L, N=N)
    replayed_model = poutine.replay(model, trace=guide_trace)
    model_trace = poutine.trace(replayed_model).get_trace(y=y)
    model_trace.compute_log_prob()
    guide_trace.compute_log_prob()
    log_prob_lhd = torch.sum(model_trace.nodes["obs"]["log_prob"], dim=2, keepdim=True)
    log_prob_prior = model_trace.nodes["z"]["log_prob"]
    log_prob_guide = guide_trace.nodes["z"]["log_prob"]
    log_iw = log_prob_lhd + log_prob_prior - log_prob_guide

    assert log_iw.shape == (S, L, 1)
    return log_iw


def generate_data(L, N, sigma_n, mu_z, sigma_z, correlated=True):
    if not correlated:
        raise NotImplementedError

    y = torch.zeros(L, N, 1)
    for l in range(L):
        z = torch.distributions.Normal(loc=mu_z, scale=sigma_z).sample()
        for n in range(N):
            y[l, n, 0] = torch.distributions.Normal(loc=z, scale=sigma_n).sample()
    return y


@torch.no_grad()
def main():
    # data
    sigma_n = 0.01
    mu_z = 1.0
    sigma_z = 0.5
    mu_z_model = 0.0
    sigma_z_model = 1.0
    sigma_n_model = sigma_n
    N_S = 25
    S = 10
    L = 32
    Ns = [1, 2, 4, 8, 16, 32, 64, 128]
    # Ns = [1, 2, 4]

    log_iw_means_no_reparam = []
    log_iw_means_reparam = []
    log_iw_stds_no_reparam = []
    log_iw_stds_reparam = []
    for N in Ns:
        y = generate_data(
            L=L,
            N=N,
            sigma_n=sigma_n,
            mu_z=mu_z,
            sigma_z=sigma_z,
        )
        for do_reparam in (True, False):
            pyro.set_rng_seed(123)
            log_iws = []
            for _ in range(N_S):
                # compute lml
                model = GlobalLVM(mu_z_model, sigma_z_model, sigma_n_model)
                guide = GlobalLVMGaussianPriorGuide(mu_z_model, sigma_z_model)
                if do_reparam:
                    log_iw = torch.zeros(S, L, N)
                    for n in range(N):
                        log_iw[:, :, n : n + 1] = compute_log_iw(
                            y=y[:, n : n + 1, :], S=S, model=model, guide=guide
                        )
                    log_iw = torch.sum(log_iw, dim=2, keepdim=True)
                else:
                    log_iw = compute_log_iw(y=y, S=S, model=model, guide=guide)
                assert log_iw.shape == (S, L, 1)

                log_iws.append(torch.sum(log_iw) / S)
            log_iws = torch.stack(log_iws)
            log_iw_mean = torch.mean(log_iws)
            log_iw_std = torch.std(log_iws)
            if do_reparam:
                log_iw_means_reparam.append(log_iw_mean)
                log_iw_stds_reparam.append(log_iw_std)
            else:
                log_iw_means_no_reparam.append(log_iw_mean)
                log_iw_stds_no_reparam.append(log_iw_std)
    log_iw_means_reparam = torch.stack(log_iw_means_reparam)
    log_iw_stds_reparam = torch.stack(log_iw_stds_reparam)
    log_iw_means_no_reparam = torch.stack(log_iw_means_no_reparam)
    log_iw_stds_no_reparam = torch.stack(log_iw_stds_no_reparam)

    # plot
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), squeeze=False)
    ax = axes[0, 0]
    ax.plot(Ns, log_iw_means_reparam, label="local reparametrization")
    ax.fill_between(
        x=Ns,
        y1=log_iw_means_reparam + log_iw_stds_reparam,
        y2=log_iw_means_reparam - log_iw_stds_reparam,
        alpha=0.2,
    )
    ax.plot(Ns, log_iw_means_no_reparam, label="no local reparametrization")
    ax.fill_between(
        x=Ns,
        y1=log_iw_means_no_reparam + log_iw_stds_no_reparam,
        y2=log_iw_means_no_reparam - log_iw_stds_no_reparam,
        alpha=0.2,
    )
    ax.grid()
    ax.legend()
    ax.set_xlabel("N")
    ax.set_ylabel("log_marginal_likelihood")
    fig.suptitle(
        f"N_S = {N_S}, L = {L}, $\sigma_n$ = {sigma_n}, $\mu_z$ = {mu_z}, $\sigma_z$ = {sigma_z}, "
        f"$\mu_z$ (model) = {mu_z_model}, $\sigma_z$ (model) = {sigma_z_model}"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
