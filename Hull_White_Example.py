#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:17:54 2023

@author: DLIU
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
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


#%%

# Assume calibrated parameters theta, kappa, sigma
theta = calibrated_params[0]  # calibrated theta
kappa = calibrated_params[1]  # calibrated kappa
sigma = calibrated_params[2]  # calibrated sigma

# Take the last available yield for the 1-month Treasury bill as the current short rate R_0
R_0 = treasury_yield_curve['DGS1MO'].iloc[-1] / 100  # Convert to decimal

# Time step for simulation
dt = 1/252  # One trading day ahead if we assume 252 trading days in a year


# Define the function to calculate zero-coupon bond price using Hull-White model
def zero_coupon_bond_price(r, kappa, theta, sigma, T):
    B = (1 - np.exp(-kappa * T)) / kappa
    A = np.exp((theta - (sigma**2) / (2 * kappa**2)) * (B - T) - (sigma**2) / (4 * kappa) * B**2)
    return A * np.exp(-B * r)

# Tenors for which we want to generate the yield curve
tenors = np.array([0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  # From 1 month to 30 years

# Initialize the current short rate
current_r = R_0

# DataFrame to store yield curves
yield_curve_df = pd.DataFrame(index=tenors)

# Simulate 5 steps ahead
for step in range(5):
    # Random shock for the Wiener process, assuming normal distribution
    dw = np.random.normal(0, np.sqrt(dt))
    
    # Discretized Hull-White process for one time step
    current_r = current_r + kappa * (theta - current_r) * dt + sigma * dw
    
    # Calculate zero-coupon bond prices for each tenor
    zcb_prices = np.array([zero_coupon_bond_price(current_r, kappa, theta, sigma, T) for T in tenors])
    
    # Calculate the yield for each tenor
    yields = -np.log(zcb_prices) / tenors
    
    # Add the yields to the DataFrame
    yield_curve_df[f'Step {step+1}'] = yields

# Plot the yield curves
for column in yield_curve_df.columns:
    plt.plot(yield_curve_df.index, yield_curve_df[column], label=column)

plt.title('Simulated Yield Curves at 5 Steps Ahead')
plt.xlabel('Tenor (years)')
plt.ylabel('Yield')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Shortest tenor for which we want to generate the yield curve
shortest_tenor = 0.0833  # Approximately 1 month

# Initialize the current short rate
current_r = R_0

# List to store the shortest tenor yields
shortest_tenor_yields = []

# Simulate 100 steps ahead
for step in range(100):
    # Random shock for the Wiener process, assuming normal distribution
    dw = np.random.normal(0, np.sqrt(dt))
    
    # Discretized Hull-White process for one time step
    current_r = current_r + kappa * (theta - current_r) * dt + sigma * dw
    
    # Calculate zero-coupon bond price for the shortest tenor
    zcb_price = zero_coupon_bond_price(current_r, kappa, theta, sigma, shortest_tenor)
    
    # Calculate the yield for the shortest tenor
    yield_ = -np.log(zcb_price) / shortest_tenor
    
    # Store the yield
    shortest_tenor_yields.append(yield_)

# Plot the distribution of the shortest tenor yields
plt.hist(shortest_tenor_yields, bins=20, density=True)
plt.title('Distribution of Forecasted Shortest Tenor Yields (100 Steps)')
plt.xlabel('Yield')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()