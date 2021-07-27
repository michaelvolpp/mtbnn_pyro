import os

import pandas as pd
import seaborn as sns
import torch
from torch import distributions as dist
from matplotlib import pyplot as plt

# logpath = os.path.join(".", "log")
# with open(os.path.join(logpath, "guide_svi_mvn"), "rb") as f:
#     guide1 = torch.load(f)
# with open(os.path.join(logpath, "guide_svi_mvn2"), "rb") as f:
#     guide2 = torch.load(f)


def gaussian_min_kl_gmm(means, covariances, weights):
    # Implements Eq. 57 from
    # https://ipvs.informatik.uni-stuttgart.de/mlr/marc/notes/gaussians.pdf

    # check shapes
    assert weights.ndim == 1
    assert means.ndim == 2
    assert covariances.ndim == 3
    n_dim = weights.shape[0]
    assert means.shape[0] == n_dim
    dim = means.shape[1]
    assert covariances.shape == (n_dim, dim, dim)

    mean = torch.einsum("ki,k->i", means, weights)
    c1 = covariances
    c2 = torch.einsum("ki,kj->kij", means, means)
    c3 = torch.einsum("i,j->ij", mean, mean)
    covariance = torch.einsum("kij,kij,k->ij", c1, c2, weights) + c3

    return mean, covariance


if __name__ == "__main__":
    # define gmm parameters
    dim = 2
    n_comp = 2
    means = torch.tensor([[-1.0], [1.0]]).expand(n_comp, dim)
    covariances = torch.stack(
        (
            torch.diag(torch.tensor([0.1 ** 2, 0.1 ** 2])),
            torch.diag(torch.tensor([0.1 ** 2, 0.1 ** 2])),
        ),
        dim=0,
    )
    weights = torch.tensor([0.5, 0.5])

    # compute normal distribution with minimal kl to gmm
    mean, covariance = gaussian_min_kl_gmm(
        means=means, covariances=covariances, weights=weights
    )

    # generate distribution from parameters
    mix = dist.Categorical(probs=weights)
    comps = dist.MultivariateNormal(loc=means, covariance_matrix=covariances)
    gmm = dist.MixtureSameFamily(mixture_distribution=mix, component_distribution=comps)
    normal = dist.MultivariateNormal(loc=mean, covariance_matrix=covariance)
    print(gmm)
    print(normal)

    # draw samples
    n_samples = 1000
    gmm_samples = gmm.sample((n_samples,))
    comp1_samples = comps[0].samples((n_samples,))
    comp2_samples = comps[1].samples((n_samples,))
    normal_samples = normal.sample((n_samples,))

    # visualize samples
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    sns.kdeplot(
        x=gmm_samples[:, 0].numpy(),
        y=gmm_samples[:, 1].numpy(),
        ax=ax,
        fill=False,
        label="GMM",
    )
    # sns.scatterplot(
    #     x=gmm_samples[:, 0].numpy(),
    #     y=gmm_samples[:, 1].numpy(),
    #     ax=ax,
    # )
    sns.kdeplot(
        x=normal_samples[:, 0].numpy(),
        y=normal_samples[:, 1].numpy(),
        ax=ax,
        fill=False,
        label="Normal",
    )
    # sns.scatterplot(
    #     x=normal_samples[:, 0].numpy(),
    #     y=normal_samples[:, 1].numpy(),
    #     ax=ax,
    # )

    ax.legend()
    ax.set_aspect("equal")
    plt.show()
