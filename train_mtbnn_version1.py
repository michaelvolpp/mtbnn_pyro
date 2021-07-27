import numpy as np
import pyro
import torch
from matplotlib import pyplot as plt
from metalearning_benchmarks import Quadratic1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

from bnn import MTBNN





if __name__ == "__main__":
    # generate data
    bm = Quadratic1D(
        n_task=8,
        n_datapoints_per_task=16,
        output_noise=0.25,
        seed_task=1234,
        seed_noise=1235,
        seed_x=1236,
    )

    # plot some tasks
    plot = False
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for task in bm:
            sort_idx = np.argsort(task.x.squeeze(), axis=0)
            x, y = task.x.squeeze()[sort_idx], task.y.squeeze()[sort_idx]
            ax.plot(x, y)
        plt.show()

    # generate BNN
    mtbnn = MTBNN(d_in=bm.d_x, d_out=bm.d_y, n_hidden=0)

    # generate optimizer
    adam = Adam({"lr": 0.01})

    # do inference
    n_iter = 100
    guide = AutoDiagonalNormal(model=mtbnn)
    svi = SVI(model=mtbnn, guide=guide, optim=adam, loss=Trace_ELBO())
    pyro.clear_param_store()
    for i in range(n_iter):
        loss = svi.step(x=torch.tensor(bm.x), y=torch.tensor(bm.y))
        if i % 50 == 0:
            print(f"[iter {i:04d}] loss = {loss:.4f}")
