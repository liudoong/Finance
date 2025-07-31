#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:57:40 2025

@author: october
"""

import numpy as np
import pandas as pd
from yahooquery import Ticker

# 下载 S&P 500 数据
ticker = Ticker('^GSPC')
data = ticker.history(start='2021-07-31', interval='1d')

# 如果是 MultiIndex（ticker, date），先 reset
if isinstance(data.index, pd.MultiIndex):
    data = data.reset_index()

# 保留收盘价，并统一处理日期
sp500 = data[['date', 'close']].copy()
sp500['date'] = pd.to_datetime(sp500['date'], utc=True).dt.tz_localize(None)
sp500.set_index('date', inplace=True)
data = sp500[::-1]


# 下载 AAPL 数据
ticker = Ticker('AAPL')
data2 = ticker.history(start='2021-07-31', interval='1d')

# 如果是 MultiIndex（ticker, date），先 reset
if isinstance(data2.index, pd.MultiIndex):
    data2 = data2.reset_index()

# 保留收盘价，并统一处理日期
aapl = data2[['date', 'close']].copy()
aapl['date'] = pd.to_datetime(aapl['date'], utc=True).dt.tz_localize(None)
aapl.set_index('date', inplace=True)
data2 = aapl[::-1]



#%%

def variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position):
    """

    Compute Potential Future Exposure 9PFE) for Variance Swap
    The definition of conditional variance swaps used in this function:

    Parameters
    ----------
    underlying_price : df
        Series of underlying asset prices
    tenor : float
        Time to maturity in year, if 6 months, input is 0.5
    mpr : int
        Day counts of margin period of risk
    quantile : float
        99% quantile, targeting on long position, if short position, it's 1-0.99
    N_vega : float
        Vega notional value
    strike : float
        The contractual strike volatility, if 20%, input 0.2 here
    position : Str
        If our posotion is long, input "long", corresponding to client's short position


    Returns
    -------
    PFE value, or a Coefficient of Vega Notional if we put N_Vega = 1
    """
    
    
    #calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(1)).dropna()
    
    #calculate annualized volatility
    rolling_realized_vol = daily_returns.rolling(window = 252).std() * np.sqrt(252)
    rolling_realized_vol = rolling_realized_vol.dropna()
    
    #calculate vol shock over MPR days
    relative_vol_shock = rolling_realized_vol / rolling_realized_vol.shift(mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()
    
    #compute quantile figures for long/short position
    quantile_shock_vol = (
        relative_vol_shock.quantile(quantile) if position == "long" else relative_vol_shock.quantile(1 - quantile)
        )
    
    #compute payoff based on shock vol
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)
    
    #adjust the annualized exposure to the tenor length
    exposure = exposure / np.sqrt(tenor)
    
    return exposure * N_vega




def barrier_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier_type, barrier_value):
    """

    Compute Potential Future Exposure 9PFE) for Conditional Variance Swap (Up-Var / Down-Var))
    The definition of conditional variance swaps used in this function:
        
        Down-Var: where realized variance is only accrued if the underlying remains below a given barrier
                  Example: 105% Down-Var accrues realized variance only when the underlying is below 105% from its initial level.
                  
        Up-Var: where realized variance is only accrued if the underlying remian above a certain barrier
    

    Parameters
    ----------
    underlying_price : df
        Series of underlying asset prices
    tenor : float
        Time to maturity in year, if 6 months, input is 0.5
    mpr : int
        Day counts of margin period of risk
    quantile : float
        99% quantile, targeting on long position, if short position, it's 1-0.99
    N_vega : float
        Vega notional value
    strike : float
        The contractual strike volatility, if 20%, input 0.2 here
    position : Str
        If our posotion is long, input "long", corresponding to client's short position
    barrier_type : Str
        "up" for Up-Var, and "down" for Down-Var
    barrier_value : float
        Barrier level as a percentage of Initial Price, e.g. 80% Up-Var puts 0.8 here, and 115% Down-Var puts 1.15 here

    Returns
    -------
    PFE value, or a Coefficient of Vega Notional if we put N_Vega = 1

    """

    #calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(1)).dropna()
    
    #select initial barrier condition
    initial_price = underlying_price.iloc[0]
    
    #construct barrier mask day by day
    if barrier_type == "up":
        valid_mask = underlying_price > initial_price * barrier_value
    elif barrier_type == "down":
        valid_mask = underlying_price < initial_price * barrier_value
    else:
        raise ValueError("Invalid barrier_type: use 'up', or 'down'  ")
    
    #apply mask on daily return (shift one day forward, so taht return match barrier)
    barrier_returns = daily_returns[valid_mask.shift(1)].dropna()
    
    #calculate rolling realized volatility (only when barrier satisfied)
    conditional_vol = barrier_returns.rolling(window = 252).std() * np.sqrt(252)
    conditional_vol = conditional_vol.dropna()
    
    
    #calculate vol shock over MPR days
    relative_vol_shock = conditional_vol / conditional_vol.shift(mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()
    
    # Quantile extraction based on position
    if position == "long":
        quantile_shock_vol = relative_vol_shock.quantile(quantile)
    elif position == "short":
        quantile_shock_vol = relative_vol_shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'long' or 'short'")

    # Approximate variance exposure payoff
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)
    exposure = exposure / np.sqrt(tenor)

    return exposure * N_vega



def corridor_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier_low, barrier_high):
    """
    Compute PFE for Corridor Variance Swap: Only accrues variance when price stays within (low, high) barrier corridor.

    Parameters
    ----------
    underlying_price : pd.Series
        Series of underlying asset prices
    tenor : float
        Time to maturity in years, e.g., 0.5 for 6 months
    mpr : int
        Margin period of risk in trading days
    quantile : float
        Quantile level (e.g., 0.99 for 99%)
    N_vega : float
        Vega notional
    strike : float
        Strike volatility (e.g., 0.2 for 20%)
    position : str
        "long" or "short"
    barrier_low : float
        Lower bound as percentage of initial price (e.g., 0.8 for 80%)
    barrier_high : float
        Upper bound as percentage of initial price (e.g., 1.2 for 120%)

    Returns
    -------
    float
        PFE value scaled by Vega Notional
    """

    # Daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(1)).dropna()

    # Initial reference price
    initial_price = underlying_price.iloc[0]

    # Build corridor mask
    within_corridor = (
        (underlying_price > initial_price * barrier_low) &
        (underlying_price < initial_price * barrier_high)
    )

    # Apply mask to returns; shift so return_t aligns with condition on price_{t-1}
    corridor_returns = daily_returns[within_corridor.shift(1)].dropna()

    # Rolling volatility in corridor
    conditional_vol = corridor_returns.rolling(window=252).std() * np.sqrt(252)
    conditional_vol = conditional_vol.dropna()

    # Volatility shock (past-looking)
    relative_vol_shock = conditional_vol / conditional_vol.shift(mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()

    # Quantile extraction based on position
    if position == "long":
        quantile_shock_vol = relative_vol_shock.quantile(quantile)
    elif position == "short":
        quantile_shock_vol = relative_vol_shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'long' or 'short'")

    # Approximate variance exposure payoff
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)
    exposure = exposure / np.sqrt(tenor)

    return exposure * N_vega



def cross_corridor_variance_swap_pfe(underlying_price, observe_price,tenor, mpr, quantile, N_vega, strike, position, barrier_low, barrier_high):
    """
    Cross-Corridor Variance Swap PFE Calculation:
    Variance accrual only occurs when the observe_price stays within a corridor,
    but variance is computed using the underlying_price.

    Parameters
    ----------
    underlying_price : pd.Series
        Series of underlying prices for realized variance computation
    observe_price : pd.Series
        Series of prices used for barrier condition (e.g., different asset)
    tenor : float
        Maturity in years (e.g., 0.5 for 6 months)
    mpr : int
        Margin period of risk (in trading days)
    quantile : float
        Quantile level for PFE (e.g., 0.99 for 99%)
    N_vega : float
        Vega notional
    strike : float
        Strike volatility (e.g., 0.2 for 20%)
    position : str
        "long" or "short"
    barrier_low : float
        Lower barrier relative to initial observe price (e.g., 0.8)
    barrier_high : float
        Upper barrier relative to initial observe price (e.g., 1.2)

    Returns
    -------
    float
        PFE value scaled by Vega Notional
    """

    # Align inputs
    underlying_price = underlying_price.copy()
    observe_price = observe_price.copy()
    underlying_price, observe_price = underlying_price.align(observe_price, join="inner")

    # Compute daily returns from underlying
    daily_returns = np.log(underlying_price / underlying_price.shift(1)).dropna()

    # Reference value for barrier (from observe_price)
    initial_observe = observe_price.iloc[0]

    # Construct mask from observe_price
    corridor_mask = (
        (observe_price > initial_observe * barrier_low) &
        (observe_price < initial_observe * barrier_high)
    )

    # Apply corridor mask to returns (shifted to match return_t with price_{t-1})
    corridor_returns = daily_returns[corridor_mask.shift(1)].dropna()

    # Rolling annualized volatility in corridor
    conditional_vol = corridor_returns.rolling(window=252).std() * np.sqrt(252)
    conditional_vol = conditional_vol.dropna()

    # Volatility shock: % change over MPR days
    relative_vol_shock = conditional_vol / conditional_vol.shift(mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()

    # Quantile shock
    if position == "long":
        quantile_shock_vol = relative_vol_shock.quantile(quantile)
    elif position == "short":
        quantile_shock_vol = relative_vol_shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'long' or 'short'")

    # Approximate variance swap payoff
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)
    exposure = exposure / np.sqrt(tenor)

    return exposure * N_vega


def cross_variance_contract_pfe(price1, price2, tenor, mpr, quantile, Notional, strike, position):
    
    """
    Compute PFE of a Cross Variance Contract:
    payoff = Notional * { (σ1^2 + σ2^2 - 2*ρ*σ1*σ2) - strike }
    
    Parameters
    ----------
    price1, price2 : pd.Series
        Historical price series of the two underlyings
    tenor : float
        Maturity in years
    mpr : int
        Margin period of risk (in trading days)
    quantile : float
        Quantile level for risk (e.g., 0.99)
    N_vega : float
        Vega notional
    strike : float
        Strike variance level
    position : str
        "long" or "short" in the variance payoff

    Returns
    -------
    float
        PFE value scaled by Vega Notional
    """
    
    #align price series
    price1, price2 = price1.align(price2, join = 'inner')
    
    # calculate daily log returns
    ret1 = np.log(price1 / price1.shift(1)).dropna()
    ret2 = np.log(price2 / price2.shift(1)).dropna()
    ret1, ret2 = ret1.align(ret2, join = 'inner')
    
    #rolling annualized std
    vol1 = ret1.rolling(252).std() * np.sqrt(252)
    vol2 = ret2.rolling(252).std() * np.sqrt(252)
    
    #Rolling correlation over 252 days
    rolling_corr = ret1.rolling(252).corr(ret2)
    
    #drop NaNs
    vol1 = vol1.dropna()
    vol2 = vol2.dropna()
    rolling_corr = rolling_corr.dropna()
    
    
    #compute cross variance at each time t
    cross_variance = vol1**2 + vol2**2 - 2 * rolling_corr * vol1 * vol2
    
    #compute change in cross variance over MPR (here use absolute shock)
    shock = cross_variance - cross_variance.shift(mpr)
    shock = shock.dropna()
    
    #select quantile
    if position == "long":
        shock_q = shock.quantile(quantile)
    elif position == "short":
        shock_q = shock.quantile(1-quantile)
    else:
        raise ValueError("Invalid position: use 'long' or 'short'  ")

    #apply shock to current cross variance
    current_cross_var = cross_variance.iloc[-1]
    shocked_cross_var = max(current_cross_var +  shock_q, 0)
    
    #compute exposure
    if position =="long":
        exposure = max(shocked_cross_var - strike, 0)
    else:
        exposure = max(strike - shocked_cross_var, 0)
    
    return exposure * Notional
    
    
    
    
    
    
    
    


#%%


# Use of Variance Swap
underlying_price = data
observe_price = aapl
tenor = 1
mpr = 10
quantile = 0.98
N_vega = 1
strike = 0.3
position = "short"

# additional for barrier var
barrier_type = "down"
barrier_value = 1.05 #e.g. 105% Down-Var

# additional for corridor var
barrier_low = 0.85 # 85% of initial price
barrier_high = 1.15 # 115% of initial price

pfe = variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position)
print(f"Potential Future Exposure 9PFE) of variance swap: {pfe}")
    
pfe = barrier_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier_type, barrier_value)
print(f"Barrier Conditional Variance Swap PFE: {pfe}")

pfe = corridor_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier_low, barrier_high)
print(f"Corridor Conditional Variance Swap PFE: {pfe}")

pfe = cross_corridor_variance_swap_pfe(underlying_price, observe_price,tenor, mpr, quantile, N_vega, strike, position, barrier_low, barrier_high)
print(f"Corridor Conditional Variance Swap PFE: {pfe}")