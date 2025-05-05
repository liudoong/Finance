#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 15:27:11 2025

@author: october
"""

# Gaussian Process Regression for Haircut Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Seed for reproducibility
np.random.seed(42)

# Manual z-score standardization
def manual_zscore(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Preprocessing function with scaling
def preprocess_data(df, target_col, feature_cols, scale_dict):
    df = df.copy()
    X_std = []
    for col in feature_cols:
        std_col = f"{col}_std"
        df[std_col] = manual_zscore(df[col]) * scale_dict.get(col, 1.0)
        X_std.append(df[std_col].values)
    X = np.column_stack(X_std)
    y = df[target_col].values
    return X, y, df

# Load and prepare the data
full_data = pd.read_csv("dataset.csv")
full_data = full_data[["Asset class_New Classes", "Final HC", "ModifiedDuration", "RatingRank"]].dropna()
full_data["Interaction"] = full_data["ModifiedDuration"] * full_data["RatingRank"]

# Filter for CLO
CLO = full_data[full_data["Asset class_New Classes"] == "CLO"].copy()
feature_cols = ["ModifiedDuration", "RatingRank", "Interaction"]
scale_dict = {"ModifiedDuration": 3.0, "RatingRank": 1.0, "Interaction": 2.0}

# Preprocess
X, y, CLO_std = preprocess_data(CLO, target_col="Final HC", feature_cols=feature_cols, scale_dict=scale_dict)

# Gaussian Process kernel setup
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)

# Fit the GP model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gp.fit(X, y)

# Example new data point for prediction
X_new = np.array([[2.5, 3, 2.5 * 3]])  # duration = 2.5, rank = 3, interaction = 7.5

# Standardize new input
X_new_std = []
for i, col in enumerate(feature_cols):
    mean = CLO[col].mean()
    std = CLO[col].std()
    X_new_std.append((X_new[0][i] - mean) / std * scale_dict[col])
X_new_std = np.array(X_new_std).reshape(1, -1)

# Predict mean and std from GP
y_pred, y_std = gp.predict(X_new_std, return_std=True)

# Posterior samples from predictive distribution
samples = np.random.normal(loc=y_pred, scale=y_std, size=1000)

# Plot predictive distribution
plt.hist(samples, bins=40, density=True)
plt.title("Posterior Predictive Distribution (via GP)")
plt.xlabel("Predicted Haircut")
plt.ylabel("Density")
plt.grid(True)
plt.show()
