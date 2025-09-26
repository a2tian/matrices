import sys
sys.path.append('../')

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from kernel import GaussianKernel
from algorithm import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy
import time


def openml_kernel(dataset, n_samples=10**4, seed=0):
    """
    Loads the specified OpenML dataset and returns a Gaussian kernel formed by 10e4 random samples, with features standardized.
    Args:
        dataset (str): The name of the OpenML dataset.
        n_samples (int): The number of samples to use for the kernel.
    """
    X, y = fetch_openml(name=dataset, return_X_y=True,
                        as_frame=False, parser="auto")
    X, y = shuffle(X, y, random_state=seed)
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


def nystrom_error(K_full, k, selector):
    """
    Computes the approximation error of a rank-k Nystrom approximation for the kernel matrix K.

    """
    start = time.time()
    # can also run partial_cholesky with K directly
    F, _ = partial_cholesky(K_full, k, selector)
    end = time.time()

    K_hat = F @ F.T

    # Compute the optimal Nystrom approximation for comparison
    # u, s, vt = scipy.sparse.linalg.svds(K_full, k=k)
    # K_opt = u @ np.diag(s) @ vt

    rel_err_f = np.linalg.norm(
        K_full - K_hat, 'fro') / np.linalg.norm(K_full, 'fro')
    rel_err_tr = np.trace(K_full - K_hat) / np.trace(K_full)

    # spec_err = np.linalg.norm(K_full - K_hat, 2)
    return {"tr": rel_err_tr, "fro": rel_err_f, "time": end - start}


def test_nystrom(datasets, n_samples, n_trials, ks, exclude=None, seed=0):
    """
    Tests the Nystrom approximation on a given dataset.

    Args:
        dataset (str): The name of the OpenML dataset.
        n_samples (int): The number of samples to use for the kernel.
        n_trials (int): The number of trials to run, per rank.
        ks (list): List of ranks for the Nystrom approximation.
    """
    if not exclude:
        exclude = []
    
    res = pd.DataFrame(columns=["dataset", "k", 
                                "rp_tr", "rp_fro", "rp_time",
                                "greedy_tr", "greedy_fro", "greedy_time", 
                                "uniform_tr", "uniform_fro", "uniform_time", 
                                "opt_tr", "opt_fro"])
    for dataset in (data_bar := tqdm(datasets)):
        data_bar.set_description(f"Dataset: {dataset}")
        K, _, _ = openml_kernel(dataset, n_samples, seed=seed)
        K_full = K[:, :]

        for k in (k_bar := tqdm(ks, leave=False)):
            k_bar.set_description(f"k={k}")
            # Compute the optimal Nystrom approximation for comparison
            
            if "opt" not in exclude:
                u, s, vt = scipy.sparse.linalg.svds(K_full, k=k)
                K_opt = u @ np.diag(s) @ vt
                opt_tr = np.trace(K_full - K_opt) / np.trace(K_full)
                opt_fro = np.linalg.norm(
                    K_full - K_opt, 'fro') / np.linalg.norm(K_full, 'fro')
            else:
                opt_tr = opt_fro = np.nan

            for i in (trial_bar := tqdm(range(n_trials), leave=False)):
                trial_bar.set_description(f"Trial {i+1}/{n_trials}")
                
                if "rp" not in exclude:
                    rp_errors = nystrom_error(K_full, k, adaptive_random)
                else:
                    rp_errors = {"tr": np.nan, "fro": np.nan, "time": np.nan}
                    
                if "greedy" not in exclude:
                    greedy_errors = nystrom_error(K_full, k, greedy)
                else:
                    greedy_errors = {"tr": np.nan, "fro": np.nan, "time": np.nan}
                    
                    
                if "uniform" not in exclude:
                    uniform_errors = nystrom_error(K_full, k, uniform)
                else:
                    uniform_errors = {"tr": np.nan, "fro": np.nan, "time": np.nan}
                
                res.loc[len(res)] = [dataset, k, 
                                     rp_errors["tr"], rp_errors["fro"], rp_errors["time"],
                                     greedy_errors["tr"], greedy_errors["fro"], greedy_errors["time"],
                                     uniform_errors["tr"], uniform_errors["fro"], uniform_errors["time"],
                                     opt_tr, opt_fro]
    return res


def generate_data(datasets, n_samples, n_trials, ks, filename, *args, **kwargs):
    res = test_nystrom(datasets, n_samples, n_trials, ks, *args, **kwargs)
    res.to_csv(filename, index=False)


def make_plots(res):
    sns.set(style="whitegrid", font_scale=2)

    # Plot frobenius error
    plt.figure(figsize=(15, 9))
    ax = sns.lineplot(data=res, x="k", y="fro", hue="dataset", marker="o")
    ax.set_yscale("log")
    ax.set_title("Nystrom Approximation Error (Frobenius Norm)")
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Relative Error")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("out/rpcholesky_fro_error.png", bbox_inches="tight")

    # Plot trace error
    plt.figure(figsize=(15, 9))
    ax = sns.lineplot(data=res, x="k", y="tr", hue="dataset", marker="o")
    ax.set_yscale("log")
    ax.set_title("Nystrom Approximation Error (Trace Norm)")
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Relative Error")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("out/rpcholesky_tr_error.png", bbox_inches="tight")

    # Plot computation time
    plt.figure(figsize=(15, 9))
    ax = sns.lineplot(data=res, x="k", y="time", hue="dataset", marker="o")
    ax.set_title("Nystrom Approximation Time")
    ax.set_xlabel("Rank (k)")
    ax.set_ylabel("Time (seconds)")
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("out/rpcholesky_time.png", bbox_inches="tight")


def make_tables(res):
    # Create summary tables
    summary = res.groupby(['dataset', 'k']).agg({
        'greedy_tr': ['median'],
        'rp_tr': ['median'],
        'opt_tr': ['median'],
        'greedy_fro': ['median'],
        'rp_fro': ['median'],
        'opt_fro': ['median']
    }).reset_index()

    print(summary)
    summary.to_latex("out/table1000.tex", index=False, float_format="%.2e")
    

def normalized_errors(df_opt, df_rel, threshold=1):
    cols = ["rp_tr", "rp_fro", "greedy_tr", "greedy_fro"]
    
    min_ks = pd.DataFrame(columns=["k"]+cols)
    for k in sorted(df_opt["k"].unique()):
        ref = df_opt.loc[df_opt["k"] == k, ["dataset", "opt_tr", "opt_fro"]]
        m = df_rel.merge(ref, on="dataset", how="inner")
        min_k = {"k": k}
        for col in cols:
            if "tr" in col:
                m[col] = m[col] / m["opt_tr_y"]
            else:
                m[col] = m[col] / m["opt_fro_y"]
            min_k[col] =  m.loc[m[col] < threshold, "k"].min()
        min_ks = pd.concat([min_ks, pd.DataFrame([min_k])], ignore_index=True)
        m = m[["dataset", "k", "rp_tr", "rp_fro", "greedy_tr", "greedy_fro"]]
        
    
    min_ks.to_csv(f"out/min_k_to_opt_r_threshold{threshold}.csv", index=False)
    min_ks = min_ks.melt(id_vars=["k"], value_vars=cols, var_name="method", value_name="min_k")

    sns.set(style="whitegrid", font_scale=2)
    plt.figure(figsize=(15, 9))
    ax = sns.lineplot(data=min_ks, x="k", y="min_k", hue="method", marker="o", linestyle="--")

    ax.set_title("Minimum Rank k to Match Optimal Rank-r Approximation")
    ax.set_xlabel("Target Rank (r)")
    ax.set_ylabel("Minimum Rank (k)")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"out/min_k_to_opt_r_threshold{threshold}.png", bbox_inches="tight")
        
        # ax = sns.lineplot(data=m, x="k", y="rp_tr", hue="dataset", marker="o")
        # ax.set_yscale("log")
        # ax.set_title("Normalized Nystrom Approximation Error (Trace Norm) - RPCholesky")
        # ax.set_xlabel("Rank (k)")
        # ax.set_ylabel("Normalized Approximation Error")
        # plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.savefig(f"out/normalized_rpcholesky_tr_error_k{k}.png", bbox_inches="tight")
        
        # plt.figure(figsize=(15, 9))
        # ax = sns.lineplot(data=m, x="k", y="rp_fro", hue="dataset", marker="o")
        # ax.set_yscale("log")
        # ax.set_title("Normalized Nystrom Approximation Error (Frobenius Norm) - RPCholesky")
        # ax.set_xlabel("Rank (k)")
        # ax.set_ylabel("Normalized Approximation Error")
        # plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.savefig(f"out/normalized_rpcholesky_fro_error_k{k}.png", bbox_inches="tight")
        


if __name__ == "__main__":
    datasets = ["yolanda", "mnist_784", "jannis",
                "volkert", "creditcard", "hls4ml_lhc_jets_hlf"]
    n_samples = 10000
    n_trials = 1
    ks = list(range(10, 101, 10))
    # ks = [10, 50] + list(range(100, 1001, 100))
    filename = "out/errors_opt_10_100.csv"

    run_experiments = False

    if run_experiments:
        res = generate_data(datasets, n_samples, n_trials, ks, filename, exclude=["uniform", "rp", "greedy"])
    else:
        res = pd.read_csv(filename)

    
    df_opt = pd.read_csv("out/errors_opt_10_100.csv")
    df_rel = pd.read_csv("out/errors_relative_rp_greedy.csv")
    normalized_errors(df_opt, df_rel)
    
    # make_plots(res)
    # make_tables(res)
