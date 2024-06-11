#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:10:36 2024

@author: dliu
"""

import numpy     as np
import yfinance  as yf
import datetime  as dt
from scipy.stats import norm

# Function to fetch data from yahoo finance

def fetch_stock_data(tickers, start, end):
    """
    This function fetch stock 'Adj Close' prices
    """
    data = yf.download(tickers, start = start, end = end)['Adj Close']
    return data

# function to calculate volatility and correlation matrix

def calculate_volatility_and_corr_matrix(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    volatilities = log_returns.std() * np.sqrt(252)
    corr_matrix = log_returns.corr().values
    return volatilities, corr_matrix


# Function to calculate the arithmetic average call option price

def arithmetic_average_option_price(S0, K, T, r, sigma, d, N, use_quasi = False):
    
    """
    S0: Most recent underlying closing price
    K:  Strike price
    T:  Time to maturity in years
    r:  Risk-free rate
    d:  Number of reset points
    N:  Number of simulation
    """
    
    dt = T / d
    discount_factor = np.exp(-r * T)
    
    if use_quasi:
        from scipy.stats.qmc import Sobol
        sobol = Sobol(d, scramble = False)
        rand_nums = sobol.random(N)
    else:
        rand_nums = np.random.rand(N, d)
        
        
    z = norm.ppf(rand_nums)
    paths = np.zeros((N, d+1))
    paths[:,0] = S0
    
    for t in range(1, d+1):
        paths[:,t] = paths[:,t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[:, t-1])
        
    arithmetic_averages = np.mean(paths[:, 1:], axis = 1)
    payoffs = np.maximum(arithmetic_averages - K, 0)
    option_price = discount_factor * np.mean(payoffs)
    
    return option_price


# Function to calculate basket call option call price

def basket_option_price(S0, K, T, r, volatilities, corr_matrix, weights, N, d):
    """
    """
    dt = T / d
    discount_factor = np.exp(-r * T)
    
    cholesky_decomp = np.linalg.cholesky(corr_matrix)
    num_assets = len(S0)
    
    paths = np.zeros((N, d + 1, num_assets))
    paths[:, 0, :] = S0
    
    for t in range(1, d + 1):
        # generate independent standard normal randome variables
        rand_nums = np.random.randn(N, num_assets)
        # create correlated random variables
        correlated_rand_nums = rand_nums @ cholesky_decomp.T
        for i in range(num_assets):
            paths[:, t, i] = paths[:, t-1, i] * np.exp((r - 0.5 * volatilities[i]**2) * dt + volatilities[i] * np.sqrt(dt) * correlated_rand_nums[:,i])
        
    #Calculate basket values
    basket_values = np.dot(paths[:, 1:, :], weights)
    arithmetic_averages = np.mean(basket_values, axis = 1)
    payoffs = np.maximum(arithmetic_averages - K, 0)
    option_price = discount_factor * np.mean(payoffs)
    
    return option_price

#%%

# Fetch a historical data sample, AAPL

ticker = "AAPL"
data   = yf.download(ticker, start = '2023-01-01', end = '2024-01-01')

# Calculate annualized volatility (sigma) and average stock price

log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1)).dropna()
sigma = log_returns.std() * np.sqrt(252) #Annualized volatility
S0 = data['Adj Close'][-1] #most recent closing price

# Option parameters

K = S0 * 1.05 # Strick price 5% above current price
T = 1 # Time to maturity in years
r = 0.05 # Risk-free rate (assumed)
d = 12 # number of reset points (monthly)
N = 100000 # number of simulation

# Calculate option pricing using Monte Carlo
option_price_mc = arithmetic_average_option_price(S0, K, T, r, sigma, d, N)
print(f"Arithmetic Average option Price (Monte Carlo): {option_price_mc}")

# Calculate option price using Quasi-Monte Carlo
option_price_qmc = arithmetic_average_option_price(S0, K , T, r, sigma, d, N, use_quasi=True)
print(f"Arithmetic Average option Price (Quasi-Monte Carlo): {option_price_qmc}")

#%%

# Define tickers and fetch data

tickers = ["AAPL", "GOOGL", "NVDA"]
data    = fetch_stock_data(tickers, start='2023-01-01', end='2024-01-01')

# calculate volatilities and correlation matrix
volatilities, corr_matrix = calculate_volatility_and_corr_matrix(data)
S0 = data.iloc[-1].values # most recent closing prices

# Operation parameters
K = np.mean(S0) * 1.05 # strike price 5% above the average current price
T = 1 # Time to maturity in years
r = 0.05 # Assumed risk-free rate
d = 12 # Number of simulations
weights = np.array([1/3, 1/3, 1/3]) # Equal weights for each stock

# calculate basket optioni price using Monte Carlo
option_price = basket_option_price(S0, K, T, r, volatilities, corr_matrix, weights, N, d)
print(f"Basket Option Price (Monte Carlo): {option_price}")



