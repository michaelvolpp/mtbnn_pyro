from functools import cache
import torch
from mtmlp.mtmlp import _get_init_wb


@cache
def cached_func(n):
    return torch.randn(n)


def main():
    # print(cached_func(1))
    # print(cached_func(2))
    # print(cached_func(3))
    # print(cached_func(1))
    # print(cached_func(2))
    # print(cached_func(3))

    print(_get_init_wb(n_tasks=2, size_w=3, size_b=4))
    print(_get_init_wb(n_tasks=2, size_w=3, size_b=4))
    print(_get_init_wb(n_tasks=3, size_w=3, size_b=4))


if __name__ == "__main__":
    main()
