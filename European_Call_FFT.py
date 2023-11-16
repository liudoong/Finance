#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:17:22 2023

@author: DLIU
"""

import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from scipy import fftpack

#%% data preparation 

# stock data download
ticker = 'AAPL'
data   = yf.download(ticker, start = '2022-01-01', end = '2023-01-01')

# calculate the historical volatility
returns  = np.log(data['Close'] / data['Close'].shift(1))
sigma    = np.std(returns.dropna()) * np.sqrt(252) # Annualize the volatility

# use the most recent closing price as the current stock price
S0 = data['Close'][-1]

# download risk-free rate data from FRED
risk_free_rate_df = web.DataReader('DGS3MO', 'fred', '2022-01-01', '2023-01-01')
r                 = risk_free_rate_df.iloc[-1,0] / 100 # convert the latest value to a decimal

#%% Black-Schole Model

# Parameters for option pricing
K = 100 # strike price
T = 1   # time to maturity in years

# Characteristic function for the Black-Scholes Model
def char_func_BS(u, T, r, sigma):
    return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma**2) * T) - 0.5 * (sigma**2) * (u**2) *T)

"""
Given the following inputs
        S0:    Current stock price
        r :    Risk-Free rate
        sigma: Volatility of the stock,
        T:     Time to maturity
        u:     Variable in the complex plane
        
In this equation
        1j:    the imaginary unit
        exp(): the exponential function
        ln():  natural logarithm
        
This characteristic function is derived udner the assumption that the logarithm of the stock price
follows a normal distribution with a mean ln(S0)+(r - 0.5 * sigma**2) * T), and variance  (sigma**2) * T   
as per the Black-Scholes model
"""

# Parameters for the FFT
N     = 2**10 # Number of points in the FFT
delta = 0.25  # Grid spacing
alpha = -1.5  # Damping factor (for numerical stability)

# FFT input array
u = np.arange(N) * delta
x = np.log(S0) + (r - 0.5 * sigma**2) * T - np.pi / delta
j = np.arange(N)
v = char_func_BS(u - (alpha + 1)*1j, T, r, sigma) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)

# apply FFT
fft_input   = np.exp(1j * u * x) * v * delta
fft_output  = fftpack.fft(fft_input)
call_prices = np.exp(-alpha * np.log(K)) / np.pi * np.real(fft_output)

# Extract the option price
strike_grid = np.exp(x + j * np.pi / (N * delta))
option_price_index = np.searchsorted(strike_grid, K)
option_price = call_prices[option_price_index]

print(f"European Call Option Price (FFT) for {ticker} under Black-Schole Model: {option_price: .2f}")

#%% Heston Model

# Heston model parameters

v0    = sigma**2 # Initial variance
kappa = 1.0      # Speed of mean reversion
theta = sigma**2 # Long run average volatility
rho   = -0.7     # Correlation coefficient between asset and volatility
xi    = 0.1      # Volatility of volatiliy

# Characteristic function for the Heston model

def char_func_Heston(u, T, r, kappa, theta, xi, rho, v0, S0):
    # Complex numbers for the calculation
    
    d = np.sqrt((kappa - 1j * rho * xi * u)**2 + (xi**2) * (1j * u + u**2))
    g = (kappa - 1j * rho * xi * u - d) / (kappa - 1j * rho * xi * u + d)
    
    C = kappa * (kappa - 1j * rho * xi * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    D = ((kappa - 1j * rho * xi * u - d) / (xi**2)) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    return np.exp(1j * u * np.log(S0) + 1j * u * r * T + D * v0 + C * theta)

# Parameters for the FFT
N     = 2**10 # Number of points in the FFT
delta = 0.25  # Grid spacing
alpha = -1.5  # Damping factor (for numerical stability)

# FFT process

u = np.arange(N) * delta
x = np.log(S0) + (r -0.5 * sigma**2) * T - np.pi / delta
j = np.arange(N)
v = char_func_Heston(u - (alpha + 1) * 1j, T, r, kappa, theta, xi, rho, v0, S0) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha +1) * u)

#apply FFT

fft_input   = np.exp(1j * u * x) * v * delta
fft_output  = fftpack.fft(fft_input)
call_price  = np.exp(-alpha * np.log(K)) / np.pi * np.real(fft_output)


# Extract option price

strike_grid = np.exp(x + j * np.pi / (N * delta))
option_price_index = np.searchsorted(strike_grid, K)
option_price = call_prices[option_price_index]

print(f"European Call Option Price (FFT) for {ticker} under Heston Model: {option_price: .2f}")


    



