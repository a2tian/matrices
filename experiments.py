from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from kernel import GaussianKernel
from algorithm import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


def openml_kernel(dataset, n_samples=10**4):
    """
    Loads the specified OpenML dataset and returns a Gaussian kernel formed by 10e4 random samples, with features standardized.
    Args:
        dataset (str): The name of the OpenML dataset.
        n_samples (int): The number of samples to use for the kernel.
    """
    X, y = fetch_openml(name=dataset, return_X_y=True,
                        as_frame=False, parser="auto")
    X, y = shuffle(X, y, random_state=0)
    X = X[:n_samples]
    y = y[:n_samples]

    # Standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = np.divide(X - X_mean, X_std, out=np.zeros_like(X,
                  dtype='float64'), where=X_std != 0)

    # Create kernel
    K = GaussianKernel(X, bandwidth=np.sqrt(X.shape[1]))

    return K, X, y


def nystrom_error(K, k, selector):
    """
    Computes the approximation error of a rank-k Nystrom approximation for the kernel matrix K.

    """
    start = time.time()
    F, _ = partial_cholesky(K, k, selector)
    end = time.time()

    K_hat = F @ F.T
    K_full = K[:, :]

    # Compute the optimal Nystrom approximation for comparison
    # u, s, vt = scipy.sparse.linalg.svds(K_full, k=k)
    # K_opt = u @ np.diag(s) @ vt

    relative_F_err = np.linalg.norm(
        K_full - K_hat, 'fro') / np.linalg.norm(K_full, 'fro')
    relative_tr_err = np.trace(K_full - K_hat) / np.trace(K_full)
    # spec_err = np.linalg.norm(K_full - K_hat, 2)
    return {"tr": relative_tr_err, "fro": relative_F_err, "time": end - start}


def test_nystrom(datasets, n_samples, n_trials, ks):
    """
    Tests the Nystrom approximation on a given dataset.

    Args:
        dataset (str): The name of the OpenML dataset.
        n_samples (int): The number of samples to use for the kernel.
        n_trials (int): The number of trials to run, per rank.
        ks (list): List of ranks for the Nystrom approximation.
    """
    res = pd.DataFrame(columns=["dataset", "k", "tr", "fro", "time"])
    for dataset in (data_bar := tqdm(datasets)):
        data_bar.set_description(f"Dataset: {dataset}")
        K, _, _ = openml_kernel(dataset, n_samples)
        for k in (k_bar := tqdm(ks, leave=False)):
            k_bar.set_description(f"k={k}")
            for _ in (trial_bar := tqdm(range(n_trials), leave=False)):
                trial_bar.set_description(f"Trial {_+1}/{n_trials}")
                errors = nystrom_error(K, k, adaptive_random)
                res.loc[len(res)] = [dataset, k, errors["tr"],
                                     errors["fro"], errors["time"]]
    return res


if __name__ == "__main__":
    datasets = ["yolanda", "mnist_784", "jannis",
                "volkert", "creditcard", "hls4ml_lhc_jets_hlf"]
    n_samples = 10000
    n_trials = 10
    ks = [10, 50] + list(range(100, 1001, 100))

    res = test_nystrom(datasets, n_samples, n_trials, ks)
    res.to_csv(f"out/rpcholesky_errors.csv", index=False)

    sns.set_theme(style="whitegrid")

    # Plot frobenius error
    plt.figure(figsize=(12, 9))
    ax = sns.lineplot(data=res, x="k", y="fro", hue="dataset", marker="o")
    ax.set_yscale("log")
    ax.set_title("Nystrom Approximation Error (Frobenius Norm)")
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Relative Error")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("out/rpcholesky_fro_error.png")

    # Plot trace error
    plt.figure(figsize=(12, 9))
    ax = sns.lineplot(data=res, x="k", y="tr", hue="dataset", marker="o")
    ax.set_yscale("log")
    ax.set_title("Nystrom Approximation Error (Trace Norm)")
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Relative Error")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("out/rpcholesky_tr_error.png")

    # Plot computation time
    plt.figure(figsize=(12, 9))
    ax = sns.lineplot(data=res, x="k", y="time", hue="dataset", marker="o")
    ax.set_yscale("log")
    ax.set_title("Nystrom Approximation Time")
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Time (seconds)")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("out/rpcholesky_time.png")
