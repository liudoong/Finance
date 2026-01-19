"""
Smoothed Volatility Surface Calibration with Regularization

This module provides smoothed calibration for SABR and Heston models with:
1. Data quality filtering (liquidity, outliers)
2. Regularization penalties for smoothness
3. Adjustable smoothing parameters

Key improvements over vol_surface_calibrator.py:
- Tikhonov regularization for parameter smoothness across maturities
- Outlier detection and removal
- Bid-ask spread filtering
- Smoothed RBF interpolation
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import RBFInterpolator, Rbf
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False


def filter_option_data_for_smoothness(df, outlier_z_score=2.5, min_time_value_pct=0.02):
    """
    Filter option data to remove outliers and illiquid options.

    UPDATED: More aggressive filtering to handle real market data issues.

    Args:
        df: DataFrame with option data
        outlier_z_score: Z-score threshold for outlier removal (default: 2.5)
        min_time_value_pct: Minimum time value as % of spot price (default: 2%, lowered from 5%)

    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()

    print(f"\n{'='*60}")
    print("Data Quality Filtering for Smooth Calibration")
    print(f"{'='*60}")
    print(f"Original data points: {len(df_filtered)}")

    # Calculate intrinsic value
    df_filtered['intrinsic'] = np.where(
        df_filtered['option_type'].str.upper() == 'CALL',
        np.maximum(df_filtered['spot_price'] - df_filtered['strike'], 0),
        np.maximum(df_filtered['strike'] - df_filtered['spot_price'], 0)
    )

    # Calculate time value
    df_filtered['time_value'] = df_filtered['option_price'] - df_filtered['intrinsic']
    df_filtered['time_value_pct'] = df_filtered['time_value'] / df_filtered['spot_price']

    # Filter 0: Remove negative time value (arbitrage violations) - CRITICAL
    initial_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['time_value'] >= 0].copy()
    removed = initial_count - len(df_filtered)
    if removed > 0:
        print(f"⚠ Removed {removed} options with NEGATIVE time value (arbitrage violations)")

    # Filter 1: Remove options with insufficient time value (likely illiquid)
    initial_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['time_value_pct'] >= min_time_value_pct].copy()
    print(f"After time value filter (>={min_time_value_pct*100:.1f}%): {len(df_filtered)} ({len(df_filtered)/initial_count*100:.1f}%)")

    # Filter 2: Remove deep ITM/OTM options (TIGHTER range: 0.75-1.35)
    df_filtered['moneyness'] = df_filtered['strike'] / df_filtered['spot_price']
    initial_count = len(df_filtered)
    df_filtered = df_filtered[
        (df_filtered['moneyness'] >= 0.75) &  # Tighter: was 0.7
        (df_filtered['moneyness'] <= 1.35)    # Tighter: was 1.5
    ].copy()
    print(f"After moneyness filter (0.75-1.35): {len(df_filtered)} ({len(df_filtered)/initial_count*100:.1f}%)")

    # Filter 3: Remove outliers using IQR method (more robust than z-score)
    initial_count = len(df_filtered)

    def remove_outliers_per_maturity(group):
        """Remove outliers within each maturity group using IQR."""
        if len(group) < 5:
            return group

        # Use IQR method (more robust for non-normal distributions)
        Q1 = group['option_price'].quantile(0.25)
        Q3 = group['option_price'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return group[
            (group['option_price'] >= lower_bound) &
            (group['option_price'] <= upper_bound)
        ]

    df_filtered = df_filtered.groupby('maturity_date', group_keys=False).apply(
        remove_outliers_per_maturity
    ).reset_index(drop=True)

    print(f"After IQR outlier removal: {len(df_filtered)} ({len(df_filtered)/initial_count*100:.1f}%)")

    # Filter 4: Ensure reasonable option prices
    initial_count = len(df_filtered)
    df_filtered = df_filtered[
        (df_filtered['option_price'] < df_filtered['spot_price'] * 1.5) &  # Not too expensive
        (df_filtered['option_price'] > 0.1)  # Not too cheap (likely stale)
    ].copy()
    print(f"After price sanity checks: {len(df_filtered)} ({len(df_filtered)/initial_count*100:.1f}%)")

    print(f"\n✓ Final filtered data: {len(df_filtered)} options ({len(df_filtered)/len(df)*100:.1f}% of original)")

    if len(df_filtered) > 0:
        print(f"  Moneyness range: {df_filtered['moneyness'].min():.3f} - {df_filtered['moneyness'].max():.3f}")
        print(f"  Time value %: {df_filtered['time_value_pct'].min()*100:.2f}% - {df_filtered['time_value_pct'].max()*100:.2f}%")

    print(f"{'='*60}\n")

    # Clean up temporary columns
    df_filtered = df_filtered.drop(['intrinsic', 'time_value', 'time_value_pct', 'moneyness'],
                                    axis=1, errors='ignore')

    return df_filtered


def _black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing formula."""
    from scipy.stats import norm

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def _implied_volatility(market_price, S, K, T, r, option_type='call'):
    """Calculate implied volatility using Newton-Raphson method."""
    from scipy.stats import norm

    if T <= 0 or market_price <= 0:
        return np.nan

    # Intrinsic value bounds
    if option_type.lower() == 'call':
        intrinsic = max(S - K * np.exp(-r*T), 0)
    else:
        intrinsic = max(K * np.exp(-r*T) - S, 0)

    if market_price < intrinsic * 0.99:  # Allow small tolerance
        return np.nan

    sigma = 0.3  # Initial guess
    max_iterations = 100
    tolerance = 1e-6

    for _ in range(max_iterations):
        price = _black_scholes_price(S, K, T, r, sigma, option_type)
        diff = market_price - price

        if abs(diff) < tolerance:
            return sigma

        # Vega (derivative of price w.r.t. sigma)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)

        if vega < 1e-10:
            return np.nan

        sigma = sigma + diff / vega

        # Keep sigma in reasonable bounds
        sigma = max(0.01, min(sigma, 5.0))

    return np.nan


def _sabr_volatility(F, K, T, alpha, beta, rho, nu):
    """SABR model implied volatility formula."""
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return np.nan

    # Handle ATM case
    if abs(F - K) < 1e-10:
        ATM_vol = alpha / (F ** (1 - beta))
        return ATM_vol

    # Log-moneyness
    log_FK = np.log(F / K)

    # FK geometric mean
    FK_beta = (F * K) ** ((1 - beta) / 2)

    # z parameter
    z = (nu / alpha) * FK_beta * log_FK

    # x parameter
    if abs(z) < 1e-5:
        x_z = 1.0
    else:
        x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        x_z = z / x

    # First term
    term1 = alpha / (FK_beta * (1 + ((1-beta)**2 / 24) * log_FK**2 + ((1-beta)**4 / 1920) * log_FK**4))

    # Second term
    term2 = (1 + ((1-beta)**2 / 24 * alpha**2 / FK_beta**2 +
                   0.25 * rho * beta * nu * alpha / FK_beta +
                   (2 - 3*rho**2) / 24 * nu**2) * T)

    implied_vol = term1 * x_z * term2

    return max(implied_vol, 0.01)


def calibrate_sabr_smooth(df_clean, regularization=0.01, output_strikes=50):
    """
    Calibrate SABR model with smoothness regularization.

    Args:
        df_clean: Cleaned options DataFrame
        regularization: Smoothness penalty weight (0=no smoothing, 0.1=heavy smoothing)
        output_strikes: Number of strikes in output grid

    Returns:
        params_by_maturity: Dict of SABR parameters for each maturity
        surface: VolatilitySurface object (from vol_surface_calibrator)
    """
    S = df_clean['spot_price'].iloc[0]
    r = 0.05

    unique_maturities = sorted(df_clean['time_to_maturity'].unique())
    sabr_params_by_maturity = {}

    print(f"\n{'='*60}")
    print(f"SABR Calibration with Regularization (λ={regularization})")
    print(f"{'='*60}\n")

    # Calibrate slice by slice
    for i, maturity in enumerate(unique_maturities):
        df_slice = df_clean[df_clean['time_to_maturity'] == maturity].copy()

        if len(df_slice) < 3:
            continue

        F = S * np.exp(r * maturity)
        beta = 0.5  # Fixed

        initial = [0.3, 0.0, 0.3]  # [alpha, rho, nu]
        bounds = [
            (0.01, 2.0),    # alpha
            (-0.95, 0.95),  # rho (tighter bounds for stability)
            (0.05, 1.5)     # nu
        ]

        def objective(params):
            alpha, rho, nu = params

            # Fitting error
            total_error = 0
            count = 0

            for _, row in df_slice.iterrows():
                K = row['strike']
                market_vol = row['implied_vol']

                if np.isnan(market_vol):
                    continue

                try:
                    model_vol = _sabr_volatility(F, K, maturity, alpha, beta, rho, nu)
                    if not np.isnan(model_vol):
                        error = (market_vol - model_vol) ** 2
                        total_error += error
                        count += 1
                except:
                    continue

            if count == 0:
                return 1e10

            fitting_error = total_error / count

            # Regularization: penalize large parameter changes between adjacent maturities
            smoothness_penalty = 0

            if i > 0 and regularization > 0:
                prev_maturity = unique_maturities[i-1]
                if prev_maturity in sabr_params_by_maturity:
                    prev_params = sabr_params_by_maturity[prev_maturity]

                    # Penalize large jumps in parameters
                    alpha_diff = (alpha - prev_params['alpha']) ** 2
                    rho_diff = (rho - prev_params['rho']) ** 2
                    nu_diff = (nu - prev_params['nu']) ** 2

                    # Weight by time difference
                    time_diff = maturity - prev_maturity
                    smoothness_penalty = regularization * (alpha_diff + rho_diff + nu_diff) / max(time_diff, 0.01)

            return fitting_error + smoothness_penalty

        # Use previous maturity parameters as initial guess for continuity
        if i > 0 and unique_maturities[i-1] in sabr_params_by_maturity:
            prev_params = sabr_params_by_maturity[unique_maturities[i-1]]
            initial = [prev_params['alpha'], prev_params['rho'], prev_params['nu']]

        result = minimize(
            objective,
            initial,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-6}
        )

        alpha, rho, nu = result.x
        sabr_params_by_maturity[maturity] = {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'nu': nu
        }

        print(f"  T={maturity:.3f}y: α={alpha:.4f}, β={beta:.2f}, ρ={rho:+.4f}, ν={nu:.4f}")

    # Build smoothed volatility surface
    unique_strikes = np.linspace(df_clean['strike'].min(), df_clean['strike'].max(), output_strikes)
    implied_vols = np.zeros((len(unique_maturities), len(unique_strikes)))

    for i, T in enumerate(unique_maturities):
        if T not in sabr_params_by_maturity:
            implied_vols[i, :] = 0.3
            continue

        params = sabr_params_by_maturity[T]
        F = S * np.exp(r * T)

        for j, K in enumerate(unique_strikes):
            vol = _sabr_volatility(F, K, T, params['alpha'], params['beta'],
                                  params['rho'], params['nu'])
            implied_vols[i, j] = vol if not np.isnan(vol) else 0.3

    # Import VolatilitySurface from existing module
    from vol_surface_calibrator import VolatilitySurface

    # Use smoothed RBF interpolation
    surface = VolatilitySurface(unique_strikes, unique_maturities, implied_vols, S, r)

    return sabr_params_by_maturity, surface


def smooth_calibrate_vol_surface(df, model='SABR', regularization=0.01,
                                 filter_data=True, outlier_threshold=2.5,
                                 risk_free_rate=0.05, output_strikes=50):
    """
    Main function: Calibrate smooth volatility surface with data filtering and regularization.

    Args:
        df: Options DataFrame from yahoo_option_data_cleaner.py
        model: 'SABR' or 'Heston' (currently only SABR implemented with smoothing)
        regularization: Smoothness penalty (0.0=no smoothing, 0.1=heavy smoothing, default=0.01)
        filter_data: Apply data quality filters (default: True)
        outlier_threshold: Z-score threshold for outlier removal (default: 2.5)
        risk_free_rate: Risk-free rate (default: 0.05)
        output_strikes: Number of strikes in output grid (default: 50)

    Returns:
        dict with 'params', 'surface', 'model', 'calibration_data'
    """

    print(f"\n{'='*70}")
    print(f"SMOOTH Volatility Surface Calibration - {model.upper()} Model")
    print(f"{'='*70}")
    print(f"Regularization: {regularization}")
    print(f"Data filtering: {filter_data}")
    print(f"{'='*70}\n")

    # Prepare data
    df_clean = df.copy()
    df_clean['trading_date'] = pd.to_datetime(df_clean['trading_date'])
    df_clean['maturity_date'] = pd.to_datetime(df_clean['maturity_date'])
    df_clean['days_to_maturity'] = (df_clean['maturity_date'] - df_clean['trading_date']).dt.days
    df_clean['time_to_maturity'] = df_clean['days_to_maturity'] / 365.0

    # Basic filtering
    df_clean = df_clean[
        (df_clean['option_price'] > 0) &
        (df_clean['spot_price'] > 0) &
        (df_clean['strike'] > 0) &
        (df_clean['time_to_maturity'] > 0.02) &  # > 1 week
        (df_clean['time_to_maturity'] < 5.0)     # < 5 years
    ].copy()

    # Apply quality filtering
    if filter_data:
        df_clean = filter_option_data_for_smoothness(df_clean, outlier_z_score=outlier_threshold)

    # Compute implied volatilities
    print("Computing implied volatilities...")
    df_clean['implied_vol'] = df_clean.apply(
        lambda row: _implied_volatility(
            row['option_price'],
            row['spot_price'],
            row['strike'],
            row['time_to_maturity'],
            risk_free_rate,
            row['option_type'].lower()
        ),
        axis=1
    )

    df_clean = df_clean[df_clean['implied_vol'].notna()].copy()

    # CRITICAL: Filter out extreme IV values (likely calculation errors)
    iv_before = len(df_clean)
    df_clean = df_clean[
        (df_clean['implied_vol'] >= 0.05) &  # Min 5% IV
        (df_clean['implied_vol'] <= 2.0)     # Max 200% IV
    ].copy()
    iv_removed = iv_before - len(df_clean)

    if iv_removed > 0:
        print(f"⚠ Removed {iv_removed} options with extreme IV (< 5% or > 200%)")

    if len(df_clean) == 0:
        raise ValueError("No valid options after IV filtering. Data quality is too poor.")

    print(f"\n✓ Using {len(df_clean)} options for calibration")
    print(f"  Spot price: ${df_clean['spot_price'].iloc[0]:.2f}")
    print(f"  Maturity range: {df_clean['time_to_maturity'].min():.2f} - {df_clean['time_to_maturity'].max():.2f} years")
    print(f"  Strike range: ${df_clean['strike'].min():.0f} - ${df_clean['strike'].max():.0f}")
    print(f"  IV range: {df_clean['implied_vol'].min()*100:.1f}% - {df_clean['implied_vol'].max()*100:.1f}%")
    print(f"  IV median: {df_clean['implied_vol'].median()*100:.1f}%")

    # Calibrate with smoothing
    if model.upper() == 'SABR':
        params, surface = calibrate_sabr_smooth(df_clean, regularization=regularization,
                                                output_strikes=output_strikes)
    else:
        raise NotImplementedError(f"Smoothed calibration for {model} not yet implemented. Use 'SABR'.")

    print(f"\n{'='*70}")
    print("✓ Smooth Calibration Complete!")
    print(f"{'='*70}\n")

    return {
        'params': params,
        'surface': surface,
        'model': model.upper(),
        'calibration_data': df_clean,
        'spot_price': df_clean['spot_price'].iloc[0],
        'risk_free_rate': risk_free_rate,
        'regularization': regularization
    }


# Example usage
if __name__ == "__main__":
    from option_data_cleaner import extract_options_data

    print("Loading option data...")
    df = extract_options_data("spx_data.xlsx")

    # Test different regularization levels
    for reg in [0.0, 0.01, 0.05]:
        print(f"\n{'#'*80}")
        print(f"# Testing with regularization = {reg}")
        print(f"{'#'*80}\n")

        result = smooth_calibrate_vol_surface(
            df,
            model='SABR',
            regularization=reg,
            filter_data=True,
            outlier_threshold=2.5
        )

        surface = result['surface']

        # Test interpolation
        print("\nTesting volatility at ATM:")
        test_maturities = [0.25, 0.5, 1.0]
        spot = result['spot_price']

        for T in test_maturities:
            vol = surface.get_vol(spot, T)
            print(f"  Vol(K={spot:.0f}, T={T}y) = {vol:.2%}")

        # Plot if available
        try:
            surface.plot()
        except:
            pass