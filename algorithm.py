import numpy as np
import math
import time
import unittest


def nystrom_approximation(A, S):
    return A[:, S] @ np.linalg.pinv(A[np.ix_(S, S)]) @ A[S, :]


def partial_cholesky(A, k, selector):
    n = A.shape[0]
    diag = A.diagonal()
    F = np.zeros((n, k))
    S = set()
    for i in range(k):
        S.add(s := selector(diag, A, F))
        g = A[:, s] - F[:, :i] @ F[s, :i].T
        F[:, i] = g / np.sqrt(g[s])
        diag = (diag - F[:, i] ** 2).clip(min=0)
        if sum(diag) == 0:
            break
    return F, S


def adaptive_random(diag, *args):
    return np.random.choice(len(diag), p=diag/sum(diag))


def greedy(diag, *args):
    return np.argmax(diag)


def uniform(diag, *args):
    return np.random.choice(np.nonzero(diag)[0])


def rpc2(diag, M, F):
    n = M.shape[0]
    p = diag/sum(diag)
    I = np.random.choice(len(diag), size=math.isqrt(n), p=p)
    D = np.diag(1/np.sqrt(p[I]))
    R = M[np.ix_(I, I)] - F[I, :] @ F[I, :].T
    S = D @ R @ D
    # S = R
    return I[best_column2(S)]
    # return np.random.choice(I, p=(w:=column_weights(S))/np.sum(w))


def best_column(S):
    S_bar = S / np.linalg.norm(S, axis=0)
    return np.argmax(np.linalg.norm(S @ S_bar, axis=0))
    

def best_column2(S):
    G = S.T @ S
    G2 = G @ G
    return np.argmax(np.diagonal(G2) / np.diagonal(G))

def column_weights(S):
    G = S.T @ S
    G2 = G @ G
    return np.diagonal(G2) / np.diagonal(G)


class Test(unittest.TestCase):
    def test_rpcholesky(self):
        """
        Check that the Nystrom approximation returned by RPCholesky is correct.
        """
        n = 1000
        A = np.random.random_sample((n, n))
        A = A @ A.T  # make random psd matrix

        F, S = partial_cholesky(A, 50, adaptive_random)
        S = list(S)
        A_hat = nystrom_approximation(A, S)
        self.assertTrue(np.allclose(A_hat, F @ F.T))

    def test_best_columns(self):
        """
        Check that best_column and best_column2 return the same result.
        """
        n = 100
        A = np.random.random_sample((n, n))
        A = A @ A.T

        start = time.perf_counter()
        i1 = best_column(A)
        end = time.perf_counter()
        print(f"best_column took {end - start:.4f} seconds")

        start = time.perf_counter()
        i2 = best_column2(A)
        end = time.perf_counter()

        print(f"best_column2 took {end - start:.4f} seconds")
        self.assertTrue(i1 == i2)


if __name__ == "__main__":
    unittest.main()
