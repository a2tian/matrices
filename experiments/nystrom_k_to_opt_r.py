#!/usr/bin/env python3

import sys
sys.path.append('../')

from algorithm import *
import pandas as pd
from data_collection import generate_data, normalized_errors
import matplotlib.pyplot as plt




if __name__ == "__main__":
    datasets = ["yolanda"]
    n_samples = 10000
    n_trials = 1
    ks = list(range(10, 1001))
    filename = "out/errors_opt_10_1000_1.csv"
    
    # res = generate_data(datasets, n_samples, n_trials, ks, filename, exclude=["uniform"])
    res = pd.read_csv(filename)
    normalized_errors(res[res["k"] <= 150], res, threshold=1)
    
    
    