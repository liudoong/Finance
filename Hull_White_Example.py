#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:17:54 2023

@author: DLIU
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy.optimize import minimize
from datetime import datetime

# Define the Hull-White model functions
def hull_white_one_factor(theta, kappa, sigma, t):
    """
    Hull-White one-factor model function.
    theta: mean-reversion level
    kappa: mean-reversion speed
    sigma: volatility
    t: time to maturity
    """
    B = (1 - np.exp(-kappa * t)) / kappa
    A = np.exp((theta - (sigma**2) / (2 * (kappa**2))) * (B - t) - (sigma**2) / (4 * kappa) * B**2)
    return A, B

def objective_function(params, market_yields, maturities):
    """
    Objective function for calibration, calculating the sum of squared errors between
    model and market yields.
    params: model parameters (theta, kappa, sigma)
    market_yields: observed market yields
    maturities: maturities corresponding to the observed yields
    """
    errors = 0
    for i, market_yield in enumerate(market_yields):
        A, B = hull_white_one_factor(*params, maturities[i])
        model_yield = -np.log(A) / maturities[i] + B * params[0]
        errors += (model_yield - market_yield)**2
    return errors

# Download US Treasury yield data from FRED
start = datetime(2020, 1, 1)
end = datetime.now()
treasury_yield_curve = web.DataReader(['DGS1MO', 'DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30'], 'fred', start, end)

# Take the last available yields as the current market yields
market_yields = treasury_yield_curve.iloc[-1].values / 100  # Convert to decimal
maturities = np.array([1/12, 1, 2, 5, 10, 30])  # Corresponding maturities in years

# Initial guess for the parameters
initial_params = [0.03, 0.1, 0.01]  # theta, kappa, sigma

# Calibrate the model
result = minimize(objective_function, initial_params, args=(market_yields, maturities), method='BFGS')

# Check if the optimization was successful
if result.success:
    calibrated_params = result.x
    print("Calibrated parameters:", calibrated_params)
else:
    raise ValueError("Calibration failed:", result.message)

# Now you can use calibrated_params to price other fixed income securities or to simulate future interest rate paths.

