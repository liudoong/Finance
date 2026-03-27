#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:35:50 2026

@author: october
"""



def variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position):
    """
    Compute Potential Future Exposure (PFE) for Variance Swap

    Parameters
    ----------
    underlying_price : pd.Series
        Series of underlying asset prices, with the most recent observation 
        on the first row and the oldest observation on the last row
    tenor : float
        Time to maturity in years, e.g., 0.5 for 6 months
    mpr : int
        Margin period of risk in trading days
    quantile : float
        Quantile level for PFE (e.g., 0.99 for 99%)
    N_vega : float
        Vega notional value
    strike : float
        Strike volatility (e.g., 0.2 for 20%)
    position : str
        "Long" or "Short"

    Returns
    -------
    float
        PFE value scaled by Vega Notional
    """

    # calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()

    # calculate annualized volatility
    rolling_realized_vol = daily_returns[::-1].rolling(window=252).std() * np.sqrt(252)
    rolling_realized_vol = rolling_realized_vol[::-1].dropna()

    # calculate vol shock over MPR days
    relative_vol_shock = rolling_realized_vol / rolling_realized_vol.shift(-mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()

    # compute quantile figures for long/short position
    if position == "Long":
        quantile_shock_vol = relative_vol_shock.quantile(quantile)
    elif position == "Short":
        quantile_shock_vol = relative_vol_shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'Long' or 'Short'")

    # compute payoff based on shock vol
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)

    # adjust the annualized exposure to the tenor length
    exposure = exposure * np.sqrt(tenor)

    return exposure * N_vega

def barrier_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier_type, barrier_value):
    """
    Compute Potential Future Exposure (PFE) for Conditional Variance Swap (Up-Var / Down-Var)
    The definition of conditional variance swaps used in this function:

    Down-Var: where realized variance is only accrued if the underlying remains below a given barrier
              Example: 105% Down-Var accrues realized variance only when the underlying is below 105% from its initial level.

    Up-Var: where realized variance is only accrued if the underlying remain above a certain barrier

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
    position : str
        If our position is Long, input "Long", if short, input "Short"
    barrier_type : str
        "up" for Up-Var, and "down" for Down-Var
    barrier_value : float
        Barrier level as a percentage of Initial Price, e.g. 80% Up-Var puts 0.8 here, and 115% Down-Var puts 1.15 here

    Returns
    -------
    PFE value, or a Coefficient of Vega Notional if we put N_Vega = 1
    """

    # calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()

    # select initial barrier condition (today's price, i.e. contract start date)
    initial_price = underlying_price.iloc[0]

    # construct barrier mask day by day
    if barrier_type == "up":
        valid_mask = underlying_price > initial_price * barrier_value
    elif barrier_type == "down":
        valid_mask = underlying_price < initial_price * barrier_value
    else:
        raise ValueError("Invalid barrier_type: use 'up', or 'down'")

    # apply mask on daily return (shift one day forward, so that return match barrier)
    barrier_returns = daily_returns[valid_mask.shift(-1)].dropna()

    # calculate rolling realized volatility (only when barrier satisfied)
    conditional_vol = barrier_returns[::-1].rolling(window=252).std() * np.sqrt(252)
    conditional_vol = conditional_vol[::-1].dropna()

    # calculate vol shock over MPR days
    relative_vol_shock = conditional_vol / conditional_vol.shift(-mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()


    if position == "Long":
        quantile_shock_vol = relative_vol_shock.quantile(quantile)
    elif position == "Short":
        quantile_shock_vol = relative_vol_shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'Long' or 'Short'")

    # approximate variance exposure payoff
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)

    exposure = exposure * np.sqrt(tenor)

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
        "Long" or "Short"
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
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()

    # Initial reference price (today's price, i.e. contract start date)
    initial_price = underlying_price.iloc[0]

    # Build corridor mask
    within_corridor = (
        (underlying_price > initial_price * barrier_low) &
        (underlying_price < initial_price * barrier_high)
    )

    # Apply mask to returns; shift so return_t aligns with condition on price_{t-1}
    corridor_returns = daily_returns[within_corridor.shift(-1)].dropna()

    # Rolling volatility in corridor
    conditional_vol = corridor_returns[::-1].rolling(window=252).std() * np.sqrt(252)
    conditional_vol = conditional_vol[::-1].dropna()

    # Volatility shock (past-looking)
    relative_vol_shock = conditional_vol / conditional_vol.shift(-mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()

    if position == "Long":
        quantile_shock_vol = relative_vol_shock.quantile(quantile)
    elif position == "Short":
        quantile_shock_vol = relative_vol_shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'Long' or 'Short'")

    # Approximate variance exposure payoff
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)

    exposure = exposure * np.sqrt(tenor)

    return exposure * N_vega


def cross_corridor_variance_swap_pfe(underlying_price, observe_price, tenor, mpr, quantile, N_vega, strike, position, barrier_low, barrier_high):
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
        "Long" or "Short"
    barrier_low : float
        Lower barrier relative to initial observe price (e.g., 0.8)
    barrier_high : float
        Upper barrier relative to initial observe price (e.g., 1.2)

    Returns
    -------
    float
        PFE value scaled by Vega Notional
    """


    underlying_price = underlying_price.squeeze().copy()
    observe_price = observe_price.squeeze().copy()
    underlying_price, observe_price = underlying_price.align(observe_price, join="inner")

    # Compute daily returns from underlying
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()

    # Reference value for barrier (today's observe price, i.e. contract start date)
    initial_observe = observe_price.iloc[0]

    # Construct mask from observe_price
    corridor_mask = (
        (observe_price > initial_observe * barrier_low) &
        (observe_price < initial_observe * barrier_high)
    )

    corridor_returns = daily_returns[corridor_mask.shift(-1)].dropna()

    # Rolling annualized volatility in corridor
    conditional_vol = corridor_returns[::-1].rolling(window=252).std() * np.sqrt(252)
    conditional_vol = conditional_vol[::-1].dropna()

    # Volatility shock: % change over MPR days
    relative_vol_shock = conditional_vol / conditional_vol.shift(-mpr) - 1
    relative_vol_shock = relative_vol_shock.dropna()

    if position == "Long":
        quantile_shock_vol = relative_vol_shock.quantile(quantile)
    elif position == "Short":
        quantile_shock_vol = relative_vol_shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'Long' or 'Short'")

    # Approximate variance swap payoff
    exposure = ((strike * 100 * (1 + quantile_shock_vol))**2 - (strike * 100)**2) / (2 * strike * 100)

    exposure = exposure * np.sqrt(tenor)

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
    Notional : float
        Notional
    strike : float
        Strike variance level
    position : str
        "Long" or "Short"

    Returns
    -------
    float
        PFE value scaled by Notional
    """

    price1 = price1.squeeze().copy()
    price2 = price2.squeeze().copy()
    price1, price2 = price1.align(price2, join='inner')

    # calculate daily log returns
    ret1 = np.log(price1 / price1.shift(-1)).dropna()
    ret2 = np.log(price2 / price2.shift(-1)).dropna()
    ret1, ret2 = ret1.align(ret2, join='inner')

    # rolling annualized std
    vol1 = ret1[::-1].rolling(252).std() * np.sqrt(252)
    vol2 = ret2[::-1].rolling(252).std() * np.sqrt(252)

    rolling_corr = ret1[::-1].rolling(252).corr(ret2[::-1])

    # flip back to descending order and drop NaNs
    vol1 = vol1[::-1].dropna()
    vol2 = vol2[::-1].dropna()
    rolling_corr = rolling_corr[::-1].dropna()

    vol1, vol2 = vol1.align(vol2, join='inner')
    vol1, rolling_corr = vol1.align(rolling_corr, join='inner')
    vol2, rolling_corr = vol2.align(rolling_corr, join='inner')

    # compute cross variance at each time t
    cross_variance = vol1**2 + vol2**2 - 2 * rolling_corr * vol1 * vol2

    # compute change in cross variance over MPR (absolute shock)
    shock = cross_variance - cross_variance.shift(-mpr)
    shock = shock.dropna()

    if position == "Long":
        shock_q = shock.quantile(quantile)
    elif position == "Short":
        shock_q = shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'Long' or 'Short'")

    current_cross_var = cross_variance.iloc[0]
    shocked_cross_var = max(current_cross_var + shock_q, 0)

    if position == "Long":
        exposure = max(shocked_cross_var - strike, 0)
    else:
        exposure = max(strike - shocked_cross_var, 0)

    exposure = exposure * np.sqrt(tenor)

    return exposure * Notional



def correlation_swap_pfe(price1, price2, window, mpr, quantile, Notional, strike, position, tenor):
    """
    Compute PFE of a Correlation swap:
    payoff = Notional * { realized_correlation - strike }

    Parameters
    ----------
    price1, price2 : pd.Series
        Historical price or FX series of the two underlyings
    window : float
        Correlation window in day count, if one year 252, if one month 21
    mpr : int
        Margin period of risk (in trading days)
    quantile : float
        Quantile level for risk (e.g., 0.99)
    Notional : float
        Notional
    strike : float
        Strike correlation level
    position : str
        "Long" or "Short"
    tenor : float
        Time to maturity in years, e.g., 0.5 for 6 months

    Returns
    -------
    float
        PFE value scaled by Notional
    """

    price1 = price1.squeeze().copy()
    price2 = price2.squeeze().copy()
    price1, price2 = price1.align(price2, join='inner')

    # calculate daily log returns
    ret1 = np.log(price1 / price1.shift(-1)).dropna()
    ret2 = np.log(price2 / price2.shift(-1)).dropna()
    ret1, ret2 = ret1.align(ret2, join='inner')

    rolling_corr = ret1[::-1].rolling(window).corr(ret2[::-1])

    rolling_corr = rolling_corr[::-1].dropna()

    # compute change in correlation over MPR (absolute shock)
    shock = rolling_corr - rolling_corr.shift(-mpr)
    shock = shock.dropna()

    if position == "Long":
        shock_q = shock.quantile(quantile)
    elif position == "Short":
        shock_q = shock.quantile(1 - quantile)
    else:
        raise ValueError("Invalid position: use 'Long' or 'Short'")

    current_corr = rolling_corr.iloc[0]

    # apply shock and bound within [-1, 1]
    shocked_corr = max(current_corr + shock_q, -1)
    shocked_corr = min(1, shocked_corr)

    if position == "Long":
        exposure = max(shocked_corr - strike, 0)
    else:
        exposure = max(strike - shocked_corr, 0)

    exposure = exposure * np.sqrt(tenor)

    return exposure * Notional