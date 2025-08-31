import numpy as np


class GaussianKernel():
    def __init__(self, data, bandwidth=1.0):
        self.data = data
        self.bandwidth = bandwidth
        self.n = data.shape[0]
        self.shape = (self.n, self.n)

    def __call__(self, i, j):
        return np.exp(-0.5 * np.linalg.norm(self.data[i] - self.data[j])**2 / (self.bandwidth**2))

    def gaussian_vec(self, vx, vy):
        """
        Computes the Gaussian kernel between two lists of points vx and vy.
        """
        ux = np.sum(vx**2, axis=1).reshape(-1, 1)
        uy = np.sum(vy**2, axis=1)
        return np.exp(-0.5 * (ux - 2 * vx @ vy.T + uy) / (self.bandwidth**2))

    def __getitem__(self, vals):
        if isinstance(vals, tuple) and len(vals) == 2:
            i, j = vals
            i = range(*i.indices(self.n)) if isinstance(i, slice) else i
            j = range(*j.indices(self.n)) if isinstance(j, slice) else j
            if isinstance(i, int) and isinstance(j, int):
                return self(i, j)
            elif isinstance(i, int):
                return self.gaussian_vec(self.data[i].reshape(1, -1), self.data)[0, :]
            elif isinstance(j, int):
                return self.gaussian_vec(self.data, self.data[j].reshape(1, -1))[:, 0]
            else:
                return self.gaussian_vec(self.data, self.data)
        else:
            raise IndexError("Invalid index. Use a tuple of two indices.")

    def __len__(self):
        return self.n

    def diagonal(self):
        return np.array([self(i, i) for i in range(self.n)])


if __name__ == "__main__":
    # GaussianKernel example
    data = np.random.rand(5, 2)
    kernel = GaussianKernel(data)

    print(f"K(0, 1) = {kernel[0, 1]}")
    print(f"K(0, :) = {kernel[0, :]}")
    print(f"K(:, 0) = {kernel[:, 0]}")
    print(f"K(:, :) = \n{kernel[:, :]}")
