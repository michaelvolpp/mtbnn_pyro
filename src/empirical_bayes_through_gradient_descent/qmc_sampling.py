import math

import torch
from matplotlib import pyplot as plt
from torch import distributions as dist
from torch.quasirandom import SobolEngine
import seaborn as sns


class QMCNormal(dist.Normal):
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        assert len(shape) >= 3, "Not implemented for empty event/batch/sample shape"
        sample_shape = shape[0]
        event_shape = shape[-1]
        batch_shape = shape[1 : len(shape) - 1]

        # sample sample_shape sobol points of dimension event_shape
        sobol_seq = SobolEngine(dimension=event_shape, scramble=True).draw(sample_shape)
        # expand to batch_shape (sobol points are the same for each draw)
        sobol_seq = sobol_seq.reshape(
            (sample_shape,) + (1,) * len(batch_shape) + (event_shape,)
        )
        sobol_seq = sobol_seq.expand((-1,) + batch_shape + (-1,))
        assert sobol_seq.shape == shape

        # use inverse trafo to get standard normal samples
        # https://botorch.org/v/0.1.4/api/_modules/botorch/sampling/qmc.html#NormalQMCEngine
        v = 0.5 + (1 - torch.finfo(sobol_seq.dtype).eps) * (sobol_seq - 0.5)
        samples_standard_normal = torch.erfinv(2 * v - 1) * math.sqrt(2)

        # transform standard normal samples to N(loc, scale**2) samples
        return self.loc + samples_standard_normal * self.scale

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def sample_n(self, n):
        raise NotImplementedError


def main():
    torch.manual_seed(123)
    # (1) visualize distribution
    mu_z = torch.tensor([[1.0]])
    scale_z = torch.tensor([[1.0]])
    qmc_normal = dist.Independent(QMCNormal(loc=mu_z, scale=scale_z), 1)
    samples = qmc_normal.rsample(sample_shape=(2 ** 3,))
    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)
    sns.kdeplot(x=samples[:, 0, 0], ax=ax)
    ax.scatter(
        x=samples[:, 0, 0], y=torch.ones_like(samples[:, 0, 0]) * 0.1, marker="x", s=10
    )
    plt.show()

    # # (2) check sampling with multiple batch shapes
    # mu_z = torch.tensor(
    #     [
    #         [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
    #         [[2.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
    #         [[3.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
    #     ]
    # )
    # scale_z = torch.tensor(
    #     [
    #         [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
    #         [[0.6, 0.5, 0.5], [1.0, 1.0, 1.0]],
    #         [[0.7, 0.5, 0.5], [1.0, 1.0, 1.0]],
    #     ]
    # )
    # qmc_normal = dist.Independent(
    #     QMCNormal(loc=mu_z, scale=scale_z), 1
    # )  # batch_shape = (3, 2), event_shape = (3,)
    # sample = qmc_normal.rsample(sample_shape=(4,))  # sample_shape = (4,)
    # log_prob = qmc_normal.log_prob(mu_z)  # log_prob_shape == batch_shape
    # assert sample.shape == (4, 3, 2, 3)
    # assert log_prob.shape == (3, 2)
    # print(sample.shape)
    # print(log_prob.shape)
    # print(sample)
    # print(log_prob)


if __name__ == "__main__":
    main()
