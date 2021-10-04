import time
import torch
from multiprocessing import cpu_count
import timeit
from matplotlib import pyplot as plt


def main():
    print(f"n_cores   = {cpu_count()}")
    print(f"n_threads = {torch.get_num_threads()}")

    now = time.time()
    for _ in range(int(1e8)):
        A = torch.rand(30000, 30000)
        A.matmul(A)
    print(f"Took {time.time() - now:.4f}s!")


def main2():
    print(f"n_cores = {cpu_count()}")

    now = time.time()
    for n in range(int(1e8)):
        n ** 2
    print(f"Took {time.time() - now:.4f}s!")


def main3():
    runtimes = []
    threads = [1] + [t for t in range(2, 49, 2)]
    for t in threads:
        torch.set_num_threads(t)
        r = timeit.timeit(
            setup="import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)",
            stmt="torch.mm(x, y)",
            number=100,
        )
        runtimes.append(r)

    plt.plot(threads, runtimes)
    plt.show()
    


if __name__ == "__main__":
    main3()
