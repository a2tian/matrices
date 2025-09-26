import numpy as np


def f(k, m, n):
    d = n-m
    x1 = np.exp(-np.sum([m/(d-i) for i in range(k)]))
    x2 = np.exp(-k*m/(d-k+1))
    x3 = np.exp(-k*m/(d))
    return x2 > x3


if __name__ == "__main__":
    print(f(1, 100, 10000))
