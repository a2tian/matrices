import numpy as np
import unittest


def nystrom_approximation(A, S):
    return A[:, S] @ np.linalg.pinv(A[np.ix_(S, S)]) @ A[S, :]


def partial_cholesky(A, k, selector):
    n = A.shape[0]
    diag = A.diagonal()
    F = np.zeros((n, k))
    S = set()
    for i in range(k):
        S.add(s := selector(diag))
        g = A[:, s] - F[:, :i] @ F[s, :i].T
        F[:, i] = g / np.sqrt(g[s])
        diag = (diag - F[:, i] ** 2).clip(min=0)
        if sum(diag) == 0:
            break
    return F, S


def adaptive_random(diag, M):
    return np.random.choice(len(diag), p=diag/sum(diag))


def greedy(diag, M):
    return np.argmax(diag)


def uniform(diag, M):
    return np.random.choice(np.nonzero(diag)[0])


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


if __name__ == "__main__":
    unittest.main()
