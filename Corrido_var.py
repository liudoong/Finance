import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Import data
data = yf.download("^GSPC", start="2020-01-01", end="2023-01-01")["Close"]
data = data[::-1]  # Reverse order directly

underlying_price = data
tenor = 0.5  # 6 months
mpr = 10  # Moving period return
quantile = 0.99
N_vega = 1
strike = 0.2
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
    
    # Define rolling window length for realized variance
    rolling_window = int(tenor * 252)
    
    # Calculate rolling realized variance (annualized)
    rolling_realized_variance = daily_returns[::-1].rolling(window=rolling_window).var() * 252
    rolling_realized_variance = rolling_realized_variance[::-1].dropna()
    
    # Calculate variance shocks over MPR days
    relative_variance_shock = rolling_realized_variance / rolling_realized_variance.shift(-mpr) - 1
    relative_variance_shock = relative_variance_shock.dropna()
    
    # Compute quantile for long/short position
    quantile_shock_rate = (
        relative_variance_shock.quantile(quantile) if position == "long" else relative_variance_shock.quantile(1 - quantile)
    )
    
    # Use rolling std dev to adjust shock impact
    realized_shock = quantile_shock_rate * daily_returns.rolling(rolling_window).std().dropna().iloc[-1]
    
    # Calculate exposure
    exposure = N_vega * (realized_shock - strike) / (2 * strike)
    
    return exposure

def variance_swap_price(underlying_price, tenor, N_vega, strike, position):
    """
    Compute Variance Swaps Price.
    
    Parameters:
    - underlying_price: Series of underlying asset prices.
    - tenor: Time to maturity in years.
    - N_vega: Vega notional.
    - strike: Variance swap strike level.
    - position: "long" or "short" variance swap.
    
    Returns:
    - Expected value for the variance swap.
    """
    
    # Calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()
    
    # Define rolling window length for realized variance
    rolling_window = int(tenor * 252)
    
    # Calculate realized variance over rolling window
    realized_variance = daily_returns[::-1].rolling(window=rolling_window).var() * 252
    realized_variance = realized_variance[::-1].dropna()
    
    # Compute variance swap payoff
    payoff = N_vega * (realized_variance - strike)
    
    # Long position benefits if realized variance > strike
    value = np.maximum(payoff, 0) if position == "long" else np.minimum(payoff, 0)
    
    return value.mean()  # Expected variance swap value



def barrier_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier_type, barrier_value):
    """
    Compute Potential Future Exposure (PFE) for Conditional Variance Swaps (UpVar/DownVar).
    
    Parameters:
    - underlying_price: Series of underlying asset prices.
    - tenor: Time to maturity in years.
    - mpr: Moving period return (days).
    - quantile: Quantile for risk exposure estimation.
    - N_vega: Vega notional.
    - strike: Variance swap strike level.
    - position: "long" or "short" variance swap.
    - barrier_type: "up" for UpVar (above barrier), "down" for DownVar (below barrier).
    - barrier_value: Barrier level as a percentage of initial price (e.g., 0.9 for DownVar).
    
    Returns:
    - PFE estimate for the conditional variance swap.
    """

    # Calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()
    
    # Define rolling window length for realized variance
    rolling_window = int(tenor * 252)
    
    # Compute realized variance over rolling window
    rolling_realized_variance = daily_returns[::-1].rolling(window=rolling_window).var() * 252
    rolling_realized_variance = rolling_realized_variance[::-1].dropna()
    
    # Determine barrier level based on initial price
    initial_price = underlying_price.iloc[-1]
    barrier_level = initial_price * barrier_value
    
    # Apply barrier condition (only accrue variance when condition is met)
    if barrier_type == "up":
        valid_mask = underlying_price > barrier_level
    elif barrier_type == "down":
        valid_mask = underlying_price < barrier_level
    else:
        raise ValueError("Invalid barrier_type. Use 'up' or 'down'.")

    # Apply the barrier condition to variance calculation
    rolling_realized_variance = rolling_realized_variance[valid_mask.shift(-rolling_window).fillna(False)]

    # Compute variance shocks over MPR days
    relative_variance_shock = rolling_realized_variance / rolling_realized_variance.shift(-mpr) - 1
    relative_variance_shock = relative_variance_shock.dropna()
    
    # Compute quantile for long/short position
    quantile_shock_rate = (
        relative_variance_shock.quantile(quantile) if position == "long" else relative_variance_shock.quantile(1 - quantile)
    )

    # Use rolling standard deviation to adjust shock impact
    realized_shock = quantile_shock_rate * daily_returns.rolling(rolling_window).std().dropna().iloc[-1]
    
    # Calculate exposure
    exposure = N_vega * (realized_shock - strike) / (2 * strike)
    
    return exposure



def corridor_variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position, barrier_low, barrier_high):
    """
    Compute Potential Future Exposure (PFE) for Corridor Variance Swaps.
    
    Parameters:
    - underlying_price: Series of underlying asset prices.
    - tenor: Time to maturity in years.
    - mpr: Margin Period of Risk.
    - quantile: Quantile for risk exposure estimation.
    - N_vega: Vega notional.
    - strike: Variance swap strike level.
    - position: "long" or "short" variance swap.
    - barrier_low: Lower barrier as a percentage of initial price (e.g., 0.9 for 90%).
    - barrier_high: Upper barrier as a percentage of initial price (e.g., 1.1 for 110%).
    
    Returns:
    - PFE estimate for the corridor variance swap.
    
    """
    # Calculate daily log return
    daily_returns = np.log(underlying_price / underlying_price.shift(-1)).dropna()
    
    # Define rolling window length for realized variance
    rolling_window = int(tenor * 252)
    
    # Compute realized variance over rolling window
    rolling_realized_variance = daily_returns[::-1].rolling(window=rolling_window).var() * 252
    rolling_realized_variance = rolling_realized_variance[::-1].dropna()
    
    # Determine barrier levels based on initial price
    initial_price = underlying_price.iloc[-1]
    lower_barrier = initial_price * barrier_low
    upper_barrier = initial_price * barrier_high
    
    # Apply corridor condition: variance accrues only when price stays within (low, high)
    valid_mask = (underlying_price > lower_barrier) & (underlying_price < upper_barrier)
    rolling_realized_variance = rolling_realized_variance[valid_mask.shift(-rolling_window).fillna(False)]
    
    # Compute variance shocks over MPR days
    relative_variance_shock = rolling_realized_variance / rolling_realized_variance.shift(-mpr) - 1
    relative_variance_shock = relative_variance_shock.dropna()
    
    # Compute quantile for long/short position
    quantile_shock_rate = (
        relative_variance_shock.quantile(quantile) if position == "long" else relative_variance_shock.quantile(1 - quantile)
    )

    # Use rolling standard deviation to adjust shock impact
    realized_shock = quantile_shock_rate * daily_returns.rolling(rolling_window).std().dropna().iloc[-1]
    
    # Calculate exposure
    exposure = N_vega * (realized_shock - strike) / (2 * strike)
    
    return exposure


# Example usage
pfe_corridor = corridor_variance_swap_pfe(
    underlying_price=underlying_price,
    tenor=tenor, 
    mpr=mpr, 
    quantile=quantile, 
    N_vega=N_vega, 
    strike=strike, 
    position=position,
    barrier_low=0.9,  # Lower barrier at 90% of initial price
    barrier_high=1.1  # Upper barrier at 110% of initial price
)

print(f"Potential Future Exposure (PFE) for Corridor Variance Swap: {pfe_corridor}")


# Example usage
pfe_upvar = barrier_variance_swap_pfe(
    underlying_price=underlying_price,
    tenor=tenor, 
    mpr=mpr, 
    quantile=quantile, 
    N_vega=N_vega, 
    strike=strike, 
    position=position,
    barrier_type="up",  # UpVar: variance accrues only if price is above barrier
    barrier_value=0.6  # 60% of initial price above variance kept
)

pfe_downvar = barrier_variance_swap_pfe(
    underlying_price=underlying_price,
    tenor=tenor, 
    mpr=mpr, 
    quantile=quantile, 
    N_vega=N_vega, 
    strike=strike, 
    position=position,
    barrier_type="down",  # DownVar: variance accrues only if price is below barrier
    barrier_value=1.6  # 160% of initial price below variance kept
)

print(f"Potential Future Exposure (PFE) for UpVar: {pfe_upvar}")
print(f"Potential Future Exposure (PFE) for DownVar: {pfe_downvar}")


# Run function
pfe = variance_swap_pfe(underlying_price, tenor, mpr, quantile, N_vega, strike, position)
price = variance_swap_price(underlying_price, tenor, N_vega, strike, position)

print(f"Potential Future Exposure (PFE): {pfe}")
print(f"Price: {price}")




  
    
