#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:09:11 2023

@author: DLIU
"""

import pandas as pd
import pandas_datareader.data as web
import numpy as np
from scipy.optimize import minimize

# Download US Treasury yield data from FRED
data = web.DataReader('TB3M', 'fred', start='2000-01-01', end='2023-10-04')
data = data['TB3M']

# Define the Hull-White model
def hull_white_discount_factor(t, tau, a, sigma):
    return np.exp(-a * tau) * np.exp(-0.5 * sigma**2 * (tau**2 - t * tau))

# Define the objective function to minimize
def objective_function(params):
    a, sigma = params
    model_discount_factors = hull_white_discount_factors(t_grid, tau_grid, a, sigma)
    error = (model_discount_factors - market_discount_factors)**2
    return np.sum(error)

# Calibrate the Hull-White model
t_grid = np.linspace(0, T, num_steps)
tau_grid = np.expand_dims(t_grid, axis=0)
market_discount_factors = np.array([1] + list(data[1:]))
initial_guess = [0.05, 0.01]

# Minimize the objective function
results = minimize(objective_function, initial_guess, method='L-BFGS-B')

# Print the calibrated parameters
a, sigma = results.x
print('Calibrated parameters:')
print('a:', a)
print('sigma:', sigma)