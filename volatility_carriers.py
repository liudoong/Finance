#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:02:56 2025

@author: october
"""

import numpy as np
import pandas as pd
import yfinance as yf

# Import data
data = yf.download("^GSPC", start="2020-01-01", end="2023-01-01")["Close"]
data = data[::-1]  # Reverse order directly

# Parameters
underlying_price = data
tenor = 0.5  # 6 months
mpr = 10  # Margin Period of Risk (days)
quantile = 0.99
N_vega = 1
strike = 0.2  # Variance swap strike level (in variance terms)
position = "long"

def variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position):
    """
    Compute Potential Future Exposure (PFE) for Variance Swaps.
    
    Parameters:
    - underlying_price: Series of underlying asset prices.
    - tenor: Time to maturity in years.
    - mpr: Margin Period of Risk.
    - quantile: Quantile for risk exposure estimation.
    - N_vega: Vega notional.
    - strike: Variance swap strike level.
    - position: "long" or "short" variance swap.
    
    Returns:
    - PFE estimate for the variance swap.
    """
    
    # Calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()
    
    # Calculate annualized volatility sequence
    rolling_realized_vol = daily_returns[::-1].rolling(window=252).std() * np.sqrt(252) * 100
    rolling_realized_vol = rolling_realized_vol[::-1].dropna()
    
    # Calculate vol shock over MPR days
    relative_vol_shock = rolling_realized_vol / rolling_realized_vol.shift(-mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()
    
    # Compute quantile figures for long/short position
    quantile_shock_vol = (
        relative_vol_shock.quantile(quantile) if position == "long" else relative_vol_shock.quantile(1 - quantile)
    )
    
    # Compute payoff based on the shock vol
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)
    
    # Adjust for tenor length
    exposure = exposure / np.sqrt(tenor)
    
    return exposure

# Run function
pfe = variance_swap_pfe(
    underlying_price=underlying_price,
    tenor=tenor, 
    mpr=mpr, 
    quantile=quantile, 
    N_vega=N_vega, 
    strike=strike, 
    position=position
)

print(f"Potential Future Exposure (PFE): {pfe}")

def conditional_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier, indicator):
    """
    Compute Potential Future Exposure (PFE) for Up and Down Variance Swaps (Volatility-Based).
    
    Parameters:
    - underlying_price: Series of underlying asset prices.
    - tenor: Time to maturity in years.
    - mpr: Margin Period of Risk.
    - quantile: Quantile for risk exposure estimation.
    - N_vega: Vega notional (set to 1 for coefficient calculation).
    - strike: Variance swap strike level (in volatility terms).
    - position: "long" or "short" variance swap.
    - barrier: Barrier level as a percentage (e.g., 5% means 1.05 * price for "up" swaps).
    - indicator: "up" for Up Variance Swap, "down" for Down Variance Swap.
    
    Returns:
    - PFE estimate for the variance swap (as a coefficient of Vega notional).
    """
    
    # Calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()
    
    # Calculate annualized rolling realized volatility (in volatility terms)
    rolling_realized_vol = daily_returns[::-1].rolling(window=252).std() * np.sqrt(252) * 100
    rolling_realized_vol = rolling_realized_vol[::-1].dropna()
    
    # Compute barrier price levels
    barrier_level = underlying_price * (1 - barrier) if indicator == "up" else underlying_price * (1 + barrier)
    
    # Filter realized volatility based on the barrier condition
    if indicator == "up":
        rolling_realized_vol = rolling_realized_vol[underlying_price > barrier_level.shift(-1)]
    elif indicator == "down":
        rolling_realized_vol = rolling_realized_vol[underlying_price < barrier_level.shift(-1)]
    
    # Compute volatility shock over MPR period
    relative_vol_shock = rolling_realized_vol / rolling_realized_vol.shift(-mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()
    
    # Compute quantile shock for long/short position
    quantile_shock_vol = (
        relative_vol_shock.quantile(quantile) if position == "long" else relative_vol_shock.quantile(1 - quantile)
    )
    
    # Compute exposure as a coefficient of Vega notional
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)
    
    # Adjust for tenor length
    exposure = exposure / np.sqrt(tenor)
    
    return exposure

# Run function for Up Variance Swap
pfe_up = conditional_variance_swap_pfe(
    underlying_price=underlying_price,
    tenor=tenor, 
    mpr=mpr, 
    quantile=quantile, 
    N_vega=N_vega, 
    strike=strike, 
    position=position,
    barrier=barrier,
    indicator="up"  # Up Variance Swap
)

# Run function for Down Variance Swap
pfe_down = conditional_variance_swap_pfe(
    underlying_price=underlying_price,
    tenor=tenor, 
    mpr=mpr, 
    quantile=quantile, 
    N_vega=N_vega, 
    strike=strike, 
    position=position,
    barrier=barrier,
    indicator="down"  # Down Variance Swap
)

print(f"Up Variance Swap PFE (as coefficient of Vega Notional): {pfe_up}")
print(f"Down Variance Swap PFE (as coefficient of Vega Notional): {pfe_down}")