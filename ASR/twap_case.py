#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 21:50:51 2026

@author: october
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def data_param(
               price: pd.DataFrame,
               volume: pd.DataFrame,
               P: float,
               f: float,
               T: float,
               rho: float,
               alpha: float,
               ) -> dict:
    """
    Assemble all input parameters required for the ASR VaR/ES models

    Parameters
    ----------
    price :  pd.DataFrame from chart, Row 0 = today (most recent), last row = oldest.
    volume : pd.DataFrame from chart, ow 0 = today (most recent), last row = oldest.
    P :      float, Prepaid amount (cash amount paid at inception).
    f :      float, Up-front delivery fraction, e.g. 0.80.
    T :      float, Execution horizon in **years**, e.g. 0.5 for 6 months.
    rho :    float, Participation rate cap, e.g. 0.10 for 10 % ADTV.
    alpha :  float, VaR/ES confidence level, e.g. 0.98 or 0.99.

    Returns
    -------
    dict with keys:
        price       – original price DataFrame (unchanged)
        volume      – original volume DataFrame (unchanged)
        P           – prepaid amount
        f           – up-front delivery fraction
        T           – execution horizon (years)
        rho         – participation rate cap
        alpha       – confidence level
        sigma       – annualised volatility estimated from log-returns
        S0          – reference price at inception (today's price, row 0)
        Q           – shares dealer must repurchase  Q = f * P / S0
        z_alpha     – standard-normal quantile at level alpha  Φ⁻¹(alpha)
        ADTV        – average daily traded volume (shares/day)
        returns     – log-return series (for diagnostics)

    Notes
    -----
    sigma estimation
        Daily log-returns  r_t = ln(S_t / S_{t-1}).
        Because the DataFrame is newest-first, we reverse before differencing
        so that r_t = ln(price[t]) - ln(price[t-1]) with time flowing forward.
        Annualised sigma = std(r_t) * sqrt(252).

    z_alpha
        Uses scipy.stats.norm.ppf(alpha), giving the standard-normal upper
        quantile.  For alpha = 0.98 this yields ≈ 2.054; for 0.99 ≈ 2.326.

    S0
        Taken from the first row of the price DataFrame (today's price).

    Q
        Approximation from Section 2.2 of the technical note: Q ≈ f * P / S0.
    """


    # Get S_0
    price_col = price.iloc[:, 0] # Series, index = Date
    S0        = float(price_col.iloc[0])


    # Calculate historical Vol \sigma using given price data
    price_sorted = price_col.iloc[::-1] # oldest first
    log_returns  = np.log(price_sorted).diff().dropna()
    sigma        = float(log_returns.std(ddof=1) * np.sqrt(252))


    # Calculate Q based on equation in section 2.2
    Q = f * P / S0


    # Prepare standard-normal quantile  Φ⁻¹(alpha)
    z_alpha = float(norm.ppf(alpha))


    # Calculate sample ADTV
    volume_col = volume.iloc[:, 0]
    ADTV       = float(volume_col.mean())


    # Assemble output dictionary
    params = {
              # raw data (pass-through)
              "price":    price,
              "volume":   volume,
              # contract parameters (pass-through)
              "P":        P,
              "f":        f,
              "T":        T,
              "rho":      rho,
              "alpha":    alpha,
              # derived scalars
              "sigma":    sigma,
              "S0":       S0,
              "Q":        Q,
              "z_alpha":  z_alpha,
              "ADTV":     ADTV,
              # diagnostic series
              "returns":  log_returns,
              }

    return params


def asr_twap(params: dict) -> float:
    """
    Compute the normalised TWAP variance V_TWAP based on section 4.3 and A1, evaluated under the 
    risk-neutral drift assumption μ = 0:
 
        V_TWAP = (2 / (T² σ²)) * [(e^{σ²T} - 1) / σ² - T] - 1
 
    The result is the variance of the normalised average price, A_TWAP / S₀, i.e. Var(S̄ / S₀) under GBM with μ = 0.
 
    Parameters
    ----------
    params : dict, which is the output of function data_param

 
    Returns
    -------
    float : V_TWAP ≥ 0.  Returns 0.0 if the analytic value is negligibly
    """
 
    sigma: float = params["sigma"]
    T:     float = params["T"]

 
    s2  = sigma ** 2       # σ²
    s2T = s2 * T           # σ²T  (the natural dimensionless scale)
 
    # ------------------------------------------------------------------ #
    # Exact formula  (Appendix A1)
    #
    #   V = (2 / (T² σ²)) * [(e^{σ²T} - 1) / σ²  -  T]  -  1
    #
    # Rewrite using s2T to keep intermediate values well-scaled:
    #
    #   bracket = (e^{s2T} - 1) / s2  -  T
    #           = T * [(e^{s2T} - 1) / s2T  -  1]
    #
    #   V = (2 / (T² s2)) * T * [(e^{s2T} - 1) / s2T - 1]  - 1
    #     = (2 / (T  s2)) *     [(e^{s2T} - 1) / s2T - 1]  - 1
    #
    # For very small s2T (< 1e-8) the exponential term loses precision;
    # switch to the Taylor series  e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24
    # which gives  (e^x - 1)/x - 1 ≈ x/2 + x²/6 + x³/24
    # and therefore  V ≈ s2T/3 + (s2T)²/12 + …
    # ------------------------------------------------------------------ #
    if s2T < 1e-8:
        # Taylor expansion to avoid catastrophic cancellation
        V = s2T / 3.0 + (s2T ** 2) / 12.0 + (s2T ** 3) / 60.0
    else:
        exp_s2T = np.exp(s2T)
        bracket = (exp_s2T - 1.0) / s2 - T          # (e^{σ²T}-1)/σ² - T
        V = (2.0 / (T ** 2 * s2)) * bracket - 1.0
 
    return float(max(V, 0.0))


def asr_VaR_ncc(params: dict, V: float) -> float:
    """
    Compute VaRα of the normalised cover cost  C/P  (Appendix A4).
 
    Parameters
    ----------
    params  : dict   Output of data_param().  Reads: f, z_alpha.
    V       : float  Output of asr_twap().    Normalised TWAP variance V.
                     OR
                     Output of asr_vwap().    Normalised VWAP variance V, but we do not have intra-day volume data, 
                                              So, we do not have this function yet. 
                                              Maybe to add it later. 
 
    Returns
    -------
    float   VaRα(C/P) — the α-quantile of the dealer's normalised cover cost.
            A value of 1.15 means there is only a (1−α) chance that the cover cost exceeds 1.15 × P.
    """
    f       = params["f"]
    z_alpha = params["z_alpha"]
 
    s2 = np.log(1.0 + V)          # A3: s² = ln(1 + V)
    s  = np.sqrt(s2)              # s  = √s²
 
    return float(f * np.exp(-0.5 * s2 + s * z_alpha))   # A4


def asr_ebp(params: dict, VaR_ncc: float) -> float:
    """
    Compute the Exposure Beyond Prepaid (A5).
 
        VaRα(L) = P · max(VaRα(C/P) − 1, 0)
 
    Parameters
    ----------
    params  : dict   Output of data_param().  Reads: P.
    VaR_ncc : float  Output of asr_VaR_ncc(). VaRα(C/P).
 
    Returns
    -------
    float   Absolute dollar loss at confidence level α. Zero when VaR_ncc ≤ 1 (cover cost stays below prepaid).
    """
    return float(params["P"] * max(VaR_ncc - 1.0, 0.0))





































# ------------------------------------------------------------------ #
# Quick self-test (runs only when the file is executed directly)
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    np.random.seed(42)
    n_days = 756                              # ~3 years of trading days
    dates  = pd.date_range(start="2023-03-22", periods=n_days, freq="B")

    # Simulate a GBM price path
    sigma_true = 0.20
    dt         = 1 / 252
    daily_ret  = np.exp(
        (0 - 0.5 * sigma_true**2) * dt
        + sigma_true * np.sqrt(dt) * np.random.randn(n_days)
    )
    prices = 100 * np.cumprod(daily_ret)      # oldest → newest
    prices_rev = prices[::-1]                 # newest → oldest (spec convention)

    price_df  = pd.DataFrame(
        {"daily_price":  prices_rev},
        index=pd.Index(dates[::-1], name="Date"),
    )
    volume_df = pd.DataFrame(
        {"daily_volume": np.random.randint(800_000, 1_200_000, size=n_days)},
        index=pd.Index(dates[::-1], name="Date"),
    )

    result = data_param(
        price   = price_df,
        volume  = volume_df,
        P       = 500_000_000,   # 5 億
        f       = 0.80,
        T       = 0.5,           # 6 months
        rho     = 0.10,
        alpha   = 0.98,
    )

    print("=== data_param output ===")
    for k, v in result.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k:10s}: DataFrame {v.shape}")
        elif isinstance(v, pd.Series):
            print(f"  {k:10s}: Series  len={len(v)}")
        else:
            print(f"  {k:10s}: {v:.6g}" if isinstance(v, float) else f"  {k:10s}: {v}")
            
            
# ------------------------------------------------------------------ #
# asr_twap test
# ------------------------------------------------------------------ #
print("\n=== asr_twap output ===")

V = asr_twap(result)
print(f"  V_TWAP (from data)      : {V:.8f}")

# Analytical check: use known sigma_true = 0.20, T = 0.5
sigma_true = 0.20
T_true     = 0.5
s2T        = sigma_true**2 * T_true
V_expected = (2.0 / (T_true**2 * sigma_true**2)) * \
             ((np.exp(s2T) - 1.0) / sigma_true**2 - T_true) - 1.0
print(f"  V_TWAP (true σ=0.20)    : {V_expected:.8f}")
print(f"  Small-σ²T approx σ²T/3  : {s2T / 3:.8f}")
print(f"  Ratio V / (σ²T/3)       : {V / (s2T / 3):.6f}  (should be ≈1 for small σ²T)")

    # ------------------------------------------------------------------ #
    # asr_VaR_ncc test
    # ------------------------------------------------------------------ #
    print("\n=== asr_VaR_ncc output ===")
 
    VaR_ncc = asr_VaR_ncc(result, V)
    s2      = np.log(1.0 + V)
    s       = np.sqrt(s2)
    print(f"  s²  = ln(1+V)           : {s2:.8f}")
    print(f"  s   = √s²               : {s:.8f}")
    print(f"  VaRα(C/P)               : {VaR_ncc:.6f}")
    print(f"  Interpretation: at {result['alpha']:.0%} confidence, cover cost ≤ "
          f"{VaR_ncc:.4f} × P  (excess over P = {max(VaR_ncc-1,0)*100:.2f}%)")
    
    
# ------------------------------------------------------------------ #
# asr_ebp test  — baseline (low vol, VaR_ncc < 1 → zero exposure)
# ------------------------------------------------------------------ #
print("\n=== asr_ebp output ===")

ebp = asr_ebp(result, VaR_ncc)
print(f"  P                       : {result['P']:,.0f}")
print(f"  VaRα(C/P)               : {VaR_ncc:.6f}")
print(f"  max(VaRα(C/P) − 1, 0)  : {max(VaR_ncc-1,0):.6f}")
print(f"  VaRα(L) = EBP           : {ebp:,.2f}")

# stress test: force VaR_ncc > 1 to show non-zero exposure
VaR_ncc_stress = 1.18
ebp_stress     = asr_ebp(result, VaR_ncc_stress)
print(f"\n  [Stress] VaRα(C/P)=1.18 → EBP = {ebp_stress:,.2f}  "
      f"({ebp_stress/result['P']*100:.1f}% of P)")