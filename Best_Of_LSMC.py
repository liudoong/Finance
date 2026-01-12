#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 20:12:10 2025
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

N_paths = 1000
T_times = 100
K = 100   # <-- FIXED: meaningful strike

# ----------------------------
# 1. Simulate S paths (GBM)
# ----------------------------
dt = 1/252
mu = 0.05
sigma = 0.2
S0 = 100

def simulate_asset():
    dW = np.random.normal(0, np.sqrt(dt), size=(N_paths, T_times))
    S = np.zeros((N_paths, T_times))
    S[:,0] = S0
    for t in range(1, T_times):
        S[:,t] = S[:,t-1] * np.exp((mu-0.5*sigma**2)*dt + sigma*dW[:,t])
    return S

S1 = simulate_asset()
S2 = simulate_asset()
S3 = simulate_asset()

# ----------------------------
# 2. True intrinsic value
# ----------------------------
intrinsic = np.maximum(K - np.maximum.reduce([S1, S2, S3]), 0)

# ----------------------------
# 3. Simulate MtM = intrinsic + noise (realistic)
# ----------------------------
mtm1 = intrinsic + np.random.normal(0, 5, size=(N_paths, T_times))
mtm2 = intrinsic + np.random.normal(0, 5, size=(N_paths, T_times))
mtm3 = intrinsic + np.random.normal(0, 5, size=(N_paths, T_times))

# ----------------------------
# 4. Initialize option value matrix
# ----------------------------
V = np.zeros((N_paths, T_times))
V[:,-1] = intrinsic[:,-1]

# ----------------------------
# 5. Backward LSMC
# ----------------------------
degree = 2

for t in range(T_times-2, -1, -1):
    X = np.stack([mtm1[:,t], mtm2[:,t], mtm3[:,t]], axis=1)

    # polynomial features
    N = X.shape[0]
    Xp = np.ones((N,1))
    Xp = np.hstack([Xp, X, X**2])
    Xp = np.hstack([Xp, (X[:,0]*X[:,1]).reshape(-1,1),
                        (X[:,0]*X[:,2]).reshape(-1,1),
                        (X[:,1]*X[:,2]).reshape(-1,1)])

    y = V[:,t+1]
    beta = np.linalg.lstsq(Xp, y, rcond=None)[0]
    
    continuation = Xp @ beta
    V[:,t] = np.maximum(continuation, intrinsic[:,t])

# ----------------------------
# 6. Exposure and PFE
# ----------------------------
exposure = np.maximum(V, 0)
pfe = np.quantile(exposure, 0.99, axis=0)
ee  = exposure.mean(axis=0)

# ----------------------------
# 7. Plot
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(ee, label="Expected Exposure")
plt.plot(pfe, label="99% PFE", linestyle="--")
plt.title("Best-of Put LSMC Exposure")
plt.legend()
plt.show()

print("Trade-level 99% PFE =", np.quantile(exposure.max(axis=1), 0.99))














