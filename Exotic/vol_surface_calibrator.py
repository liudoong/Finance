"""
Volatility Surface Calibration for Heston and SABR Models

This module calibrates stochastic volatility models (Heston/SABR) to option market data
and constructs standardized volatility surfaces compatible with QuantLib.

Usage:
    from vol_surface_calibrator import calibrate_vol_surface

    df = extract_options_data("data.xlsx")  # from yahoo_option_data_cleaner
    surface = calibrate_vol_surface(df, model='Heston')

    # Use in QuantLib
    vol_surface = surface.to_quantlib()
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import RBFInterpolator, griddata
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    warnings.warn("QuantLib not available. QuantLib export will be disabled.")


class VolatilitySurface:
    """
    Standardized volatility surface with interpolation/extrapolation capabilities.
    Compatible with QuantLib for use in diffusion processes.
    """

    def __init__(self, strikes, maturities, implied_vols, spot_price, risk_free_rate=0.05):
        """
        Args:
            strikes: Array of strike prices
            maturities: Array of time to maturity (in years)
            implied_vols: 2D array of implied volatilities [maturity x strike]
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
        """
        self.strikes = np.array(strikes)
        self.maturities = np.array(maturities)
        self.implied_vols = np.array(implied_vols)
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate

        # Create interpolator for smooth surface
        self._build_interpolator()

    def _build_interpolator(self):
        """Build RBF interpolator for the volatility surface."""
        # Create mesh grid points
        points = []
        values = []

        for i, maturity in enumerate(self.maturities):
            for j, strike in enumerate(self.strikes):
                if not np.isnan(self.implied_vols[i, j]):
                    # Use log-moneyness and sqrt(time) for better interpolation
                    log_moneyness = np.log(strike / self.spot_price)
                    sqrt_time = np.sqrt(maturity)
                    points.append([log_moneyness, sqrt_time])
                    values.append(self.implied_vols[i, j])

        if len(points) > 0:
            self.interpolator = RBFInterpolator(
                np.array(points),
                np.array(values),
                kernel='thin_plate_spline',
                smoothing=0.0
            )
        else:
            self.interpolator = None

    def get_vol(self, strike, maturity):
        """
        Get interpolated implied volatility for any strike/maturity.

        Args:
            strike: Strike price (can be array)
            maturity: Time to maturity in years (can be array)

        Returns:
            Implied volatility (or array of vols)
        """
        if self.interpolator is None:
            return np.nan

        strike = np.atleast_1d(strike)
        maturity = np.atleast_1d(maturity)

        log_moneyness = np.log(strike / self.spot_price)
        sqrt_time = np.sqrt(maturity)

        points = np.column_stack([log_moneyness, sqrt_time])
        vols = self.interpolator(points)

        # Apply reasonable bounds
        vols = np.clip(vols, 0.01, 3.0)

        return vols[0] if len(vols) == 1 else vols

    def to_quantlib(self, calendar=None, day_counter=None):
        """
        Convert to QuantLib BlackVarianceSurface for use in diffusion processes.

        Returns:
            QuantLib.BlackVarianceSurface object
        """
        if not QUANTLIB_AVAILABLE:
            raise ImportError("QuantLib is not installed")

        if calendar is None:
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        if day_counter is None:
            day_counter = ql.Actual365Fixed()

        # Reference date (today)
        today = ql.Date.todaysDate()

        # Convert maturities to QuantLib dates
        # Ensure maturities are sorted and unique
        unique_maturities = np.unique(self.maturities)

        # Create date-maturity pairs and ensure unique dates
        date_maturity_pairs = []
        seen_dates = set()

        for tau in unique_maturities:
            days = int(tau * 365)
            exp_date = today + ql.Period(days, ql.Days)

            # If this date already exists, increment by 1 day until unique
            while exp_date in seen_dates:
                days += 1
                exp_date = today + ql.Period(days, ql.Days)

            seen_dates.add(exp_date)
            date_maturity_pairs.append((exp_date, tau))

        # Sort by date to ensure QuantLib requirement
        date_maturity_pairs.sort(key=lambda x: x[0].serialNumber())

        expiration_dates = [pair[0] for pair in date_maturity_pairs]
        sorted_maturities = [pair[1] for pair in date_maturity_pairs]

        # Create variance surface (variance = vol^2)
        variances = self.implied_vols ** 2

        # Create mapping from sorted maturities to original indices
        maturity_indices = {mat: idx for idx, mat in enumerate(self.maturities)}

        # Create QuantLib matrix
        # QuantLib expects: rows = strikes, columns = dates
        # Our data is: rows = maturities, columns = strikes
        # So we need to transpose
        matrix = ql.Matrix(len(self.strikes), len(sorted_maturities))
        for i in range(len(self.strikes)):
            for j, tau in enumerate(sorted_maturities):
                orig_j = maturity_indices[tau]
                matrix[i][j] = variances[orig_j, i] if not np.isnan(variances[orig_j, i]) else 0.0

        # Create BlackVarianceSurface
        variance_surface = ql.BlackVarianceSurface(
            today,
            calendar,
            expiration_dates,
            list(self.strikes),
            matrix,
            day_counter
        )

        return variance_surface

    def plot(self, figsize=(12, 5)):
        """Plot the volatility surface (both 3D and 2D slice views)."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib not available for plotting")
            return

        fig = plt.figure(figsize=figsize)

        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')

        # Create dense grid for smooth plotting
        K_grid = np.linspace(self.strikes.min(), self.strikes.max(), 50)
        T_grid = np.linspace(self.maturities.min(), self.maturities.max(), 50)
        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

        # Interpolate volatilities
        V_mesh = np.zeros_like(K_mesh)
        for i in range(len(T_grid)):
            V_mesh[i, :] = self.get_vol(K_grid, T_grid[i])

        ax1.plot_surface(K_mesh, T_mesh, V_mesh, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity (years)')
        ax1.set_zlabel('Implied Volatility')
        ax1.set_title('Volatility Surface')

        # 2D slice plot (vol smile for different maturities)
        ax2 = fig.add_subplot(122)

        for maturity in [0.25, 0.5, 1.0, 2.0]:
            if maturity <= self.maturities.max() and maturity >= self.maturities.min():
                vols = self.get_vol(K_grid, maturity)
                moneyness = K_grid / self.spot_price
                ax2.plot(moneyness, vols, label=f'T={maturity}y')

        ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
        ax2.set_xlabel('Moneyness (K/S)')
        ax2.set_ylabel('Implied Volatility')
        ax2.set_title('Volatility Smile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


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


def _implied_volatility(market_price, S, K, T, r, option_type='call', initial_guess=0.3):
    """Calculate implied volatility using Newton-Raphson method."""
    from scipy.stats import norm

    if T <= 0 or market_price <= 0:
        return np.nan

    # Intrinsic value bounds
    if option_type.lower() == 'call':
        intrinsic = max(S - K * np.exp(-r*T), 0)
    else:
        intrinsic = max(K * np.exp(-r*T) - S, 0)

    if market_price < intrinsic:
        return np.nan

    sigma = initial_guess
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


def _heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, option_type='call'):
    """
    Heston model option pricing using characteristic function.

    Parameters:
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of volatility
        rho: Correlation between asset and volatility
    """
    if not QUANTLIB_AVAILABLE:
        # Fallback: use approximate BS with average volatility
        avg_vol = np.sqrt((v0 + theta) / 2)
        return _black_scholes_price(S, K, T, r, avg_vol, option_type)

    # Use QuantLib for accurate Heston pricing
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()

    today = ql.Date.todaysDate()
    maturity_date = today + ql.Period(int(T * 365), ql.Days)

    # Market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, day_count)
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, day_count)
    )

    # Heston process
    heston_process = ql.HestonProcess(
        rate_handle, dividend_handle, spot_handle,
        v0, kappa, theta, sigma, rho
    )

    # Option
    option_type_ql = ql.Option.Call if option_type.lower() == 'call' else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_type_ql, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    # Pricing engine
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
    option.setPricingEngine(engine)

    return option.NPV()


def _sabr_volatility(F, K, T, alpha, beta, rho, nu):
    """
    SABR model implied volatility formula.

    Parameters:
        F: Forward price
        K: Strike price
        T: Time to maturity
        alpha: Initial volatility level
        beta: CEV parameter (0=normal, 1=lognormal)
        rho: Correlation
        nu: Volatility of volatility
    """
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


def _calibrate_heston(df_clean, initial_params=None, output_strikes=50):
    """
    Calibrate Heston model to market option prices.

    Args:
        output_strikes: Number of strikes in output surface grid

    Returns:
        params: dict with v0, kappa, theta, sigma_v, rho
        surface: VolatilitySurface object
    """
    S = df_clean['spot_price'].iloc[0]
    r = 0.05  # Risk-free rate (can be adjusted)

    # Initial parameters [v0, kappa, theta, sigma_v, rho]
    if initial_params is None:
        initial_params = [0.04, 2.0, 0.04, 0.3, -0.5]

    # Bounds for parameters
    bounds = [
        (0.001, 0.5),   # v0 (initial variance)
        (0.1, 10.0),    # kappa (mean reversion speed)
        (0.001, 0.5),   # theta (long-term variance)
        (0.01, 2.0),    # sigma_v (vol of vol)
        (-0.99, 0.99)   # rho (correlation)
    ]

    # Objective function: minimize squared pricing errors
    def objective(params):
        v0, kappa, theta, sigma_v, rho = params

        # Feller condition: 2*kappa*theta > sigma_v^2
        if 2 * kappa * theta < sigma_v**2:
            return 1e10

        total_error = 0
        count = 0

        for _, row in df_clean.iterrows():
            K = row['strike']
            T = row['time_to_maturity']
            market_price = row['option_price']
            option_type = row['option_type'].lower()

            try:
                model_price = _heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type)
                error = (market_price - model_price) ** 2
                total_error += error
                count += 1
            except:
                continue

        if count == 0:
            return 1e10

        return total_error / count

    # Optimize using differential evolution (global optimization)
    print("Calibrating Heston model...")
    result = differential_evolution(
        objective,
        bounds,
        maxiter=200,
        popsize=15,
        tol=1e-4,
        seed=42,
        workers=1,
        updating='deferred'
    )

    v0, kappa, theta, sigma_v, rho = result.x

    params = {
        'v0': v0,
        'kappa': kappa,
        'theta': theta,
        'sigma_v': sigma_v,
        'rho': rho,
        'calibration_error': result.fun
    }

    print(f"Heston Parameters:")
    print(f"  v0 (initial variance): {v0:.6f}")
    print(f"  kappa (mean reversion): {kappa:.4f}")
    print(f"  theta (long-term variance): {theta:.6f}")
    print(f"  sigma_v (vol of vol): {sigma_v:.4f}")
    print(f"  rho (correlation): {rho:.4f}")
    print(f"  Calibration RMSE: {np.sqrt(result.fun):.4f}")

    # Build volatility surface using calibrated model
    unique_maturities = sorted(df_clean['time_to_maturity'].unique())
    unique_strikes = np.linspace(df_clean['strike'].min(), df_clean['strike'].max(), output_strikes)

    implied_vols = np.zeros((len(unique_maturities), len(unique_strikes)))

    for i, T in enumerate(unique_maturities):
        for j, K in enumerate(unique_strikes):
            # Price using Heston
            call_price = _heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, 'call')
            # Back out implied vol
            iv = _implied_volatility(call_price, S, K, T, r, 'call')
            implied_vols[i, j] = iv if not np.isnan(iv) else 0.3

    surface = VolatilitySurface(unique_strikes, unique_maturities, implied_vols, S, r)

    return params, surface


def _calibrate_sabr(df_clean, initial_params=None, output_strikes=50):
    """
    Calibrate SABR model to market implied volatilities.
    Uses maturity-slice calibration (separate parameters for each maturity).

    Args:
        output_strikes: Number of strikes in output surface grid

    Returns:
        params: dict with alpha, beta, rho, nu for each maturity
        surface: VolatilitySurface object
    """
    S = df_clean['spot_price'].iloc[0]
    r = 0.05

    unique_maturities = sorted(df_clean['time_to_maturity'].unique())

    # Store SABR parameters for each maturity
    sabr_params_by_maturity = {}

    print("Calibrating SABR model (slice-by-slice)...")

    for maturity in unique_maturities:
        df_slice = df_clean[df_clean['time_to_maturity'] == maturity].copy()

        if len(df_slice) < 3:
            continue

        # Forward price (approximate)
        F = S * np.exp(r * maturity)

        # Initial parameters [alpha, rho, nu] with beta=0.5 fixed
        beta = 0.5  # Fixed for stability

        if initial_params is None:
            initial = [0.3, 0.0, 0.3]
        else:
            initial = initial_params

        bounds = [
            (0.001, 2.0),   # alpha
            (-0.99, 0.99),  # rho
            (0.01, 2.0)     # nu
        ]

        def objective(params):
            alpha, rho, nu = params

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

            return total_error / count if count > 0 else 1e10

        # Optimize
        result = minimize(
            objective,
            initial,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200}
        )

        alpha, rho, nu = result.x
        sabr_params_by_maturity[maturity] = {
            'alpha': alpha,
            'beta': beta,
            'rho': rho,
            'nu': nu
        }

        print(f"  T={maturity:.2f}y: α={alpha:.4f}, β={beta:.2f}, ρ={rho:.4f}, ν={nu:.4f}")

    # Build volatility surface
    unique_strikes = np.linspace(df_clean['strike'].min(), df_clean['strike'].max(), output_strikes)
    implied_vols = np.zeros((len(unique_maturities), len(unique_strikes)))

    for i, T in enumerate(unique_maturities):
        if T not in sabr_params_by_maturity:
            implied_vols[i, :] = 0.3  # Default
            continue

        params = sabr_params_by_maturity[T]
        F = S * np.exp(r * T)

        for j, K in enumerate(unique_strikes):
            vol = _sabr_volatility(F, K, T, params['alpha'], params['beta'],
                                  params['rho'], params['nu'])
            implied_vols[i, j] = vol if not np.isnan(vol) else 0.3

    surface = VolatilitySurface(unique_strikes, unique_maturities, implied_vols, S, r)

    return sabr_params_by_maturity, surface


def calibrate_vol_surface(df, model='Heston', risk_free_rate=0.05, min_maturity_days=7, output_strikes=50):
    """
    Main function to calibrate volatility surface from options data.

    Args:
        df: DataFrame from yahoo_option_data_cleaner.py with columns:
            - underlying_ticker, trading_date, maturity_date, strike,
              option_type, option_price, spot_price
        model: 'Heston' or 'SABR'
        risk_free_rate: Risk-free interest rate (default: 0.05)
        min_maturity_days: Minimum days to maturity to include (default: 7)
        output_strikes: Number of strikes in output surface grid (default: 50)

    Returns:
        dict with:
            - 'params': Calibrated model parameters
            - 'surface': VolatilitySurface object
            - 'model': Model name
            - 'calibration_data': Cleaned DataFrame used for calibration
    """

    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    required_cols = ['trading_date', 'maturity_date', 'strike', 'option_type',
                     'option_price', 'spot_price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    model = model.upper()
    if model not in ['HESTON', 'SABR']:
        raise ValueError("Model must be 'Heston' or 'SABR'")

    # Data cleaning and preparation
    print(f"\n{'='*60}")
    print(f"Volatility Surface Calibration - {model} Model")
    print(f"{'='*60}")

    df_clean = df.copy()

    # Convert dates
    df_clean['trading_date'] = pd.to_datetime(df_clean['trading_date'])
    df_clean['maturity_date'] = pd.to_datetime(df_clean['maturity_date'])

    # Calculate time to maturity (in years)
    df_clean['days_to_maturity'] = (df_clean['maturity_date'] - df_clean['trading_date']).dt.days
    df_clean['time_to_maturity'] = df_clean['days_to_maturity'] / 365.0

    # Filter data
    df_clean = df_clean[
        (df_clean['option_price'] > 0) &
        (df_clean['spot_price'] > 0) &
        (df_clean['strike'] > 0) &
        (df_clean['time_to_maturity'] > min_maturity_days / 365.0) &
        (df_clean['time_to_maturity'] < 5.0)  # Max 5 years
    ].copy()

    if len(df_clean) == 0:
        raise ValueError("No valid data after filtering")

    # Calculate implied volatilities for SABR
    if model == 'SABR':
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

        # Remove failed IV calculations
        df_clean = df_clean[df_clean['implied_vol'].notna()].copy()

    print(f"Using {len(df_clean)} option contracts for calibration")
    print(f"Spot price: {df_clean['spot_price'].iloc[0]:.2f}")
    print(f"Maturity range: {df_clean['time_to_maturity'].min():.2f} - {df_clean['time_to_maturity'].max():.2f} years")
    print(f"Strike range: {df_clean['strike'].min():.0f} - {df_clean['strike'].max():.0f}")

    # Calibrate model
    if model == 'HESTON':
        params, surface = _calibrate_heston(df_clean, output_strikes=output_strikes)
    else:  # SABR
        params, surface = _calibrate_sabr(df_clean, output_strikes=output_strikes)

    print(f"\n{'='*60}")
    print("Calibration complete!")
    print(f"{'='*60}\n")

    return {
        'params': params,
        'surface': surface,
        'model': model,
        'calibration_data': df_clean,
        'spot_price': df_clean['spot_price'].iloc[0],
        'risk_free_rate': risk_free_rate
    }


# Example usage
if __name__ == "__main__":
    from yahoo_option_data_cleaner import extract_options_data

    # Load data
    print("Loading option data...")
    df = extract_options_data("spx_infvol_20260109.xlsx")

    # Test both models
    for model_name in ['Heston', 'SABR']:
        print(f"\n\n{'#'*70}")
        print(f"# Testing {model_name} Model")
        print(f"{'#'*70}\n")

        result = calibrate_vol_surface(df, model=model_name)

        # Access results
        surface = result['surface']
        params = result['params']

        # Test interpolation
        print("\nTesting volatility interpolation:")
        test_strikes = [6000, 6500, 7000]
        test_maturity = 1.0  # 1 year

        for K in test_strikes:
            vol = surface.get_vol(K, test_maturity)
            print(f"  Vol(K={K}, T={test_maturity}y) = {vol:.2%}")

        # Try QuantLib export
        if QUANTLIB_AVAILABLE:
            try:
                ql_surface = surface.to_quantlib()
                print(f"\n✓ Successfully exported to QuantLib surface")
            except Exception as e:
                print(f"\n✗ QuantLib export failed: {e}")

        # Plot (if matplotlib available)
        try:
            surface.plot()
        except:
            print("Plotting not available")