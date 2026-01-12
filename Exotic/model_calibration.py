"""
Model Calibration Module for Heston and Hull-White Models
==========================================================

This module handles the calibration of:
1. Heston stochastic volatility model (for equity)
2. Hull-White interest rate model (for rates)

Run this module independently to calibrate models and save parameters.
The calibrated parameters can then be imported by the main simulation module.

Usage:
    python model_calibration.py

Output:
    - Prints calibrated parameters to console
    - Returns calibrated HestonModelParams and HullWhiteModelParams objects
"""

try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import time


# =============================================================================
# 0. RATE LIMITING AND RETRY UTILITIES
# =============================================================================

class RateLimiter:
    """
    Simple rate limiter to control request frequency to avoid API rate limits.

    Yahoo Finance may return "Too Many Requests" error if we make requests too quickly.
    This class ensures minimum delay between API calls.
    """
    def __init__(self, min_interval: float = 2.0):
        """
        Args:
            min_interval: Minimum seconds between requests (default 2.0 seconds)
        """
        self.min_interval = min_interval
        self.last_request_time = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            print(f"  ⏱ Rate limiting: waiting {wait_time:.1f}s before next request...")
            time.sleep(wait_time)

        self.last_request_time = time.time()


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 5.0):
    """
    Retry a function with exponential backoff if it fails.

    Useful for handling temporary network issues or rate limit errors.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts (default 3)
        initial_delay: Initial delay in seconds (doubles each retry)

    Returns:
        Result of func() if successful

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if it's a rate limit error
            if 'rate limit' in error_str or 'too many requests' in error_str or '429' in error_str:
                if attempt < max_retries - 1:
                    print(f"  ⚠ Rate limit error (attempt {attempt + 1}/{max_retries})")
                    print(f"  ⏱ Waiting {delay:.0f}s before retry...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"  ✗ Failed after {max_retries} attempts due to rate limiting")
                    raise
            else:
                # For non-rate-limit errors, fail immediately
                raise

    raise last_exception


# Global rate limiter (shared across all API calls)
_rate_limiter = RateLimiter(min_interval=2.0)


# =============================================================================
# 1. MARKET DATA INPUT MODULE
# =============================================================================

@dataclass
class EquityMarketData:
    """
    Market data for equity underlying (SPX in this case).
    """
    # Spot data
    spot_price: float  # Current SPX level

    # Volatility surface data (for calibration)
    atm_volatility: float  # ATM implied volatility (historical vol)
    vol_surface: Optional[Dict[tuple, float]] = None  # (strike, tenor) -> vol

    # Historical data for calibration
    historical_prices: Optional[List[float]] = None  # Time series for calibration

    # Repo rate / borrow cost
    repo_rate: float = 0.0  # Equity repo rate (usually small)

    # Dividend yield (ignored per requirement, but included for completeness)
    dividend_yield: float = 0.0

    @classmethod
    def from_yahoo(cls, ticker: str, lookback_days: int = 756, repo_rate: float = 0.0):
        """
        Fetch equity market data from Yahoo Finance with rate limiting and retry logic.

        Args:
            ticker: Stock ticker (e.g., "^GSPC" for SPX, "SPY" for SPY ETF)
            lookback_days: Days of historical data for volatility calculation
                          (default 756 = 3 years, recommended for Heston calibration)
            repo_rate: Equity repo rate (default 0.0, usually small for indices)

        Returns:
            EquityMarketData instance with current spot and historical volatility
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Please install: pip install yfinance")

        print(f"✓ Fetching data from Yahoo Finance for {ticker}...")

        def fetch_data():
            """Inner function for retry logic"""
            # Apply rate limiting
            _rate_limiter.wait()

            # Download data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=f"{lookback_days}d")

            if hist_data.empty:
                raise ValueError(f"No data found for ticker {ticker}")

            return hist_data

        # Fetch with retry logic
        hist_data = retry_with_backoff(fetch_data, max_retries=3, initial_delay=5.0)

        # Get spot price (most recent close)
        spot_price = float(hist_data['Close'].iloc[-1])

        # Calculate historical volatility (annualized)
        returns = hist_data['Close'].pct_change().dropna()
        historical_vol = float(returns.std() * np.sqrt(252))  # Annualized volatility

        # Get historical prices as list
        historical_prices = hist_data['Close'].tolist()

        print(f"  Spot Price: {spot_price:,.2f}")
        print(f"  Historical Vol: {historical_vol:.2%}")
        print(f"  Historical Prices: {len(historical_prices)} days")

        return cls(
            spot_price=spot_price,
            atm_volatility=historical_vol,
            vol_surface=None,
            historical_prices=historical_prices,
            repo_rate=repo_rate,
            dividend_yield=0.0
        )

    @classmethod
    def from_excel(cls, excel_file: str, spot_price: float,
                   historical_prices_file: Optional[str] = None,
                   repo_rate: float = 0.0):
        """
        Load equity market data from local Excel file containing option data.

        This method reads implied volatility data from an Excel file (e.g., downloaded
        SPX options data) and builds a volatility surface for Heston calibration.

        Args:
            excel_file: Path to Excel file with option data (columns: Strike, Implied Volatility, etc.)
            spot_price: Current spot price of the underlying (manually provided)
            historical_prices_file: Optional path to CSV/Excel with historical prices for vol estimation
                                   If None, will use ATM implied vol as historical vol estimate
            repo_rate: Equity repo rate (default 0.0)

        Returns:
            EquityMarketData instance with spot, implied vol surface, and historical data

        Example Excel format:
            Contract Name | Strike | Implied Volatility | Bid | Ask | ...
            SPXW260112P02800000 | 2800 | 2.7031 | 0.0 | 0.05 | ...
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Please install: pip install pandas openpyxl")

        print(f"✓ Loading option data from Excel: {excel_file}")

        # Read Excel file
        df = pd.read_excel(excel_file)

        # Validate required columns
        required_cols = ['Strike', 'Implied Volatility']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Excel file must contain '{col}' column. Found: {df.columns.tolist()}")

        # Clean data - remove NaN volatilities and convert to decimal (if in percentage)
        df = df.dropna(subset=['Implied Volatility'])

        # Check if IV is in percentage (>1) or decimal format
        if df['Implied Volatility'].mean() > 1.0:
            print("  Converting implied volatilities from percentage to decimal format")
            df['Implied Volatility'] = df['Implied Volatility'] / 100.0

        # Build volatility surface: (strike, tenor) -> implied_vol
        # For single-expiry data, we'll use a simple tenor label
        vol_surface = {}

        # Extract expiry from contract name if available (e.g., SPXW260112P... -> 260112)
        if 'Contract Name' in df.columns:
            # Try to extract date from contract name (format: SPXW260112P...)
            sample_contract = df['Contract Name'].iloc[0]
            if len(sample_contract) >= 12:
                try:
                    date_str = sample_contract[4:10]  # Extract YYMMDD
                    from datetime import datetime
                    expiry_date = datetime.strptime('20' + date_str, '%Y%m%d')
                    days_to_expiry = (expiry_date - datetime.now()).days
                    tenor = f"{days_to_expiry}D"
                    print(f"  Detected expiry: {expiry_date.strftime('%Y-%m-%d')} ({days_to_expiry} days)")
                except:
                    tenor = "1M"  # Default if parsing fails
            else:
                tenor = "1M"
        else:
            tenor = "1M"  # Default tenor if no contract name

        # Populate vol surface
        for _, row in df.iterrows():
            strike = float(row['Strike'])
            iv = float(row['Implied Volatility'])

            # Sanity check on IV (should be between 1% and 300%)
            if 0.01 <= iv <= 3.0:
                vol_surface[(strike, tenor)] = iv

        print(f"  Loaded {len(vol_surface)} volatility points")

        # Calculate ATM volatility (use strikes near spot price)
        atm_tolerance = 0.05  # Within 5% of spot
        near_atm_vols = []

        for (strike, t), vol in vol_surface.items():
            moneyness = strike / spot_price
            if abs(moneyness - 1.0) < atm_tolerance:
                near_atm_vols.append(vol)

        if near_atm_vols:
            atm_volatility = float(np.median(near_atm_vols))
            print(f"  ATM Volatility: {atm_volatility:.2%} (from {len(near_atm_vols)} near-ATM options)")
        else:
            # Use median of all IVs as fallback
            atm_volatility = float(np.median([v for v in vol_surface.values()]))
            print(f"  ATM Volatility: {atm_volatility:.2%} (median of all options)")

        # Load historical prices if provided
        historical_prices = None
        if historical_prices_file:
            print(f"  Loading historical prices from: {historical_prices_file}")
            try:
                if historical_prices_file.endswith('.csv'):
                    hist_df = pd.read_csv(historical_prices_file)
                else:
                    hist_df = pd.read_excel(historical_prices_file)

                # Assume first column is date, second is price (or look for 'Close' column)
                if 'Close' in hist_df.columns:
                    historical_prices = hist_df['Close'].dropna().tolist()
                elif len(hist_df.columns) >= 2:
                    historical_prices = hist_df.iloc[:, 1].dropna().tolist()

                print(f"  Loaded {len(historical_prices)} historical prices")
            except Exception as e:
                print(f"  Warning: Could not load historical prices: {e}")
                historical_prices = None

        print(f"  Spot Price: {spot_price:,.2f}")
        print(f"  Volatility Surface: {len(vol_surface)} points")

        # Print sample of vol surface
        print(f"  Sample volatility surface:")
        sample_points = sorted(vol_surface.items(), key=lambda x: abs(x[0][0] - spot_price))[:5]
        for (strike, t), vol in sample_points:
            moneyness = strike / spot_price
            print(f"    Strike {strike:,.0f} ({moneyness:>6.1%} moneyness), {t}: {vol:>6.2%}")

        return cls(
            spot_price=spot_price,
            atm_volatility=atm_volatility,
            vol_surface=vol_surface,
            historical_prices=historical_prices,
            repo_rate=repo_rate,
            dividend_yield=0.0
        )


@dataclass
class RatesMarketData:
    """
    Interest rate market data for discounting and simulation.
    USD rates for SPX options.
    """
    # Discount curve data
    curve_date: datetime
    tenor_points: List[str]  # e.g., ["1D", "1W", "1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"]
    zero_rates: List[float]  # Zero rates corresponding to tenor points

    # Day count convention
    day_count: str = "ACT/360"  # Common for USD

    # Calendar
    calendar: str = "UnitedStates"  # US calendar for SPX

    @classmethod
    def from_fred(cls, currency: str = "USD", date: Optional[datetime] = None):
        """
        Fetch interest rate curve from FRED (Federal Reserve Economic Data) - free.

        Args:
            currency: Currency code (currently only "USD" supported)
            date: Curve date (defaults to today)

        Returns:
            RatesMarketData instance with US Treasury zero curve
        """
        if currency != "USD":
            raise ValueError("Currently only USD rates are supported via FRED")

        try:
            import pandas_datareader as pdr
        except ImportError:
            raise ImportError("Please install pandas_datareader: pip install pandas-datareader")

        # FRED series codes for US Treasury rates
        fred_series = {
            "1M": "DGS1MO",   # 1-Month Treasury
            "3M": "DGS3MO",   # 3-Month Treasury
            "6M": "DGS6MO",   # 6-Month Treasury
            "1Y": "DGS1",     # 1-Year Treasury
            "2Y": "DGS2",     # 2-Year Treasury
            "5Y": "DGS5",     # 5-Year Treasury
            "10Y": "DGS10",   # 10-Year Treasury
            "30Y": "DGS30"    # 30-Year Treasury
        }

        tenor_points = []
        zero_rates = []

        # Fetch most recent available data (last 30 days)
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        print(f"✓ Fetching USD rates from FRED (last 30 days)...")

        for tenor, series_code in fred_series.items():
            try:
                # Apply rate limiting for FRED as well
                _rate_limiter.wait()

                # Fetch data from FRED
                data = pdr.get_data_fred(series_code, start=start_date, end=end_date)

                if not data.empty and not data.iloc[-1].isna().all():
                    rate = float(data.iloc[-1].values[0]) / 100.0  # Convert from percentage
                    tenor_points.append(tenor)
                    zero_rates.append(rate)
                    print(f"  {tenor:>4s}: {rate:>6.2%}")
                else:
                    print(f"  {tenor:>4s}: No data available")
            except Exception as e:
                print(f"  {tenor:>4s}: Error fetching - {str(e)}")

        if not tenor_points:
            raise ValueError("Failed to fetch any rate data from FRED")

        print(f"✓ Successfully fetched {len(tenor_points)} tenor points")

        return cls(
            curve_date=date or datetime.now(),
            tenor_points=tenor_points,
            zero_rates=zero_rates,
            day_count="ACT/360",
            calendar="UnitedStates"
        )


# =============================================================================
# 2. HESTON MODEL CALIBRATION
# =============================================================================

@dataclass
class HestonModelParams:
    """
    Heston stochastic volatility model parameters for equity simulation.

    dS_t = r S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa(theta - v_t)dt + sigma sqrt(v_t) dW^v_t
    dW^S_t dW^v_t = rho dt
    """
    # Initial variance
    v0: float  # Initial variance (vol^2)

    # Long-term variance
    theta: float  # Long-term variance level

    # Mean reversion speed
    kappa: float  # Speed of mean reversion

    # Volatility of volatility
    sigma: float  # Vol of vol

    # Correlation
    rho: float  # Correlation between stock and vol processes

    # Calibration source
    calibrated: bool = False
    calibration_instruments: Optional[List] = None

    @classmethod
    def estimate_from_vol_surface(cls, equity_data: EquityMarketData):
        """
        Estimate Heston parameters from implied volatility surface (when historical prices unavailable).

        This is a simplified estimation method using the volatility surface shape:
        - v0: ATM variance
        - theta: ATM variance (assume current = long-term for simplicity)
        - kappa: Estimated from vol surface steepness
        - sigma: Estimated from vol smile curvature
        - rho: Estimated from put-call skew

        Args:
            equity_data: EquityMarketData with vol_surface populated

        Returns:
            HestonModelParams with estimated parameters

        Note:
            This is a rough estimation. For production, use proper calibration
            algorithms (e.g., minimize difference between model and market prices).
        """
        if equity_data.vol_surface is None or len(equity_data.vol_surface) == 0:
            raise ValueError("Need volatility surface for estimation. "
                           "Use estimate_from_historical() if you have historical prices instead.")

        print(f"✓ Estimating Heston parameters from volatility surface:")
        print(f"  Using {len(equity_data.vol_surface)} volatility points")

        # Extract strikes and vols
        strikes = [k for k, t in equity_data.vol_surface.keys()]
        vols = [equity_data.vol_surface[(k, t)] for k, t in equity_data.vol_surface.keys()]

        spot = equity_data.spot_price

        # 1. Initial variance v0: Use ATM volatility
        v0 = equity_data.atm_volatility ** 2
        print(f"  v0 from ATM vol: {v0:.6f} (vol={np.sqrt(v0):.2%})")

        # 2. Long-term variance theta: Use ATM as well (simplified)
        theta = v0
        print(f"  θ (set equal to v0): {theta:.6f}")

        # 3. Correlation rho: Estimate from skew (OTM put vol vs OTM call vol)
        # Negative rho creates negative skew (higher OTM put vol)
        otm_put_strikes = [k for k in strikes if k < spot * 0.95]  # More than 5% OTM puts
        otm_call_strikes = [k for k in strikes if k > spot * 1.05]  # More than 5% OTM calls

        if otm_put_strikes and otm_call_strikes:
            # Get average vol for OTM puts and calls
            otm_put_vols = [equity_data.vol_surface[(k, t)] for k, t in equity_data.vol_surface.keys()
                           if k in otm_put_strikes]
            otm_call_vols = [equity_data.vol_surface[(k, t)] for k, t in equity_data.vol_surface.keys()
                            if k in otm_call_strikes]

            put_vol_avg = np.mean(otm_put_vols) if otm_put_vols else equity_data.atm_volatility
            call_vol_avg = np.mean(otm_call_vols) if otm_call_vols else equity_data.atm_volatility

            # Skew = put vol - call vol (typically positive for equities)
            skew = put_vol_avg - call_vol_avg

            # Rough mapping: stronger skew -> more negative rho
            # Typical equity: skew ~ 0.03-0.10, rho ~ -0.5 to -0.8
            rho = np.clip(-0.5 - skew * 3.0, -0.9, -0.3)
            print(f"  ρ from skew: {rho:.4f} (skew={skew:.2%})")
        else:
            rho = -0.7  # Default for equities
            print(f"  ρ (default): {rho:.4f}")

        # 4. Volatility of volatility (sigma): Estimate from vol curvature
        # Higher curvature -> higher vol of vol
        if len(strikes) >= 3:
            # Calculate variance of implied vols as proxy for vol of vol
            vol_variance = np.var(vols)
            # Rough mapping: vol_variance to sigma
            sigma = np.clip(np.sqrt(vol_variance) * 3.0, 0.1, 1.0)
            print(f"  σ from vol curvature: {sigma:.4f}")
        else:
            sigma = 0.3  # Default
            print(f"  σ (default): {sigma:.4f}")

        # 5. Mean reversion kappa: Harder to estimate from single-expiry surface
        # Use typical equity value
        kappa = 2.0
        print(f"  κ (typical equity): {kappa:.4f}")

        print()
        print(f"✓ Estimated Heston parameters (from vol surface):")
        print(f"  v0 (initial var):  {v0:.6f}  (vol={np.sqrt(v0):.2%})")
        print(f"  θ (long-term var): {theta:.6f}  (vol={np.sqrt(theta):.2%})")
        print(f"  κ (mean reversion):{kappa:.4f}")
        print(f"  σ (vol of vol):    {sigma:.4f}")
        print(f"  ρ (correlation):   {rho:.4f}")
        print()
        print("  Note: This is a rough estimation from vol surface.")
        print("  For production, use proper calibration to option prices.")

        return cls(
            v0=v0,
            theta=theta,
            kappa=kappa,
            sigma=sigma,
            rho=rho,
            calibrated=True,
            calibration_instruments=[f"Vol surface: {len(equity_data.vol_surface)} points"]
        )

    @classmethod
    def estimate_from_historical(cls, equity_data: EquityMarketData):
        """
        Estimate Heston parameters from historical price data (hybrid method).

        Hybrid approach:
        - v0 (initial variance): Estimated from recent 3 months for current market state
        - Other parameters: Estimated from full 3-year dataset for long-term characteristics

        Args:
            equity_data: EquityMarketData with historical_prices populated
                        Recommended: 3 years of data (756 trading days)

        Returns:
            HestonModelParams with estimated parameters
        """
        if equity_data.historical_prices is None or len(equity_data.historical_prices) < 252:
            raise ValueError("Need at least 252 days (1 year) of historical prices for estimation. "
                           "Recommended: 756 days (3 years) for better long-term parameter estimates.")

        prices = np.array(equity_data.historical_prices)
        total_days = len(prices)

        # Calculate returns
        returns = np.diff(np.log(prices))

        # 1. Initial variance v0: use recent 3 months for CURRENT market state
        recent_returns = returns[-63:]  # Last 3 months (approx 63 trading days)
        v0 = float(np.var(recent_returns) * 252)  # Annualized variance

        print(f"✓ Estimating Heston parameters (Hybrid Method):")
        print(f"  - v0: using recent 63 days (3 months)")
        print(f"  - θ, κ, σ, ρ: using full {total_days} days ({total_days/252:.1f} years)")
        print()

        # 2. Long-term variance theta: use FULL SAMPLE for long-term characteristics
        theta = float(np.var(returns) * 252)  # Annualized long-term variance

        # 3. Mean reversion speed kappa: estimate from volatility autocorrelation
        # Use full sample for better estimation of mean reversion
        # Calculate rolling volatility (30-day windows)
        window = 30
        rolling_var = []
        for i in range(len(returns) - window):
            window_var = np.var(returns[i:i+window]) * 252
            rolling_var.append(window_var)

        rolling_var = np.array(rolling_var)

        # Estimate mean reversion from AR(1) on rolling variance
        if len(rolling_var) > 1:
            lagged_var = rolling_var[:-1]
            current_var = rolling_var[1:]

            # Simple regression: v_t = a + b*v_{t-1}
            # kappa ≈ -log(b) * 252 (annualized)
            b = np.corrcoef(lagged_var, current_var)[0, 1]
            kappa = float(-np.log(max(b, 0.01)) * 252 / window)
            kappa = np.clip(kappa, 0.5, 5.0)  # Reasonable range
        else:
            kappa = 2.0  # Default

        # 4. Vol of vol sigma: estimate from volatility of rolling volatility
        # Use full sample for better estimation
        sigma_vol = float(np.std(rolling_var))
        sigma = float(sigma_vol / np.sqrt(theta) * 2.0)  # Approximation
        sigma = np.clip(sigma, 0.1, 1.0)  # Reasonable range

        # 5. Correlation rho: correlation between returns and volatility changes
        # Use full sample for better estimation of leverage effect
        # In equity markets, typically negative (leverage effect)
        if len(rolling_var) > 1:
            # Align returns with volatility changes
            ret_aligned = returns[window-1:-1]  # Returns at time t
            vol_changes = np.diff(rolling_var)  # Change in variance from t-1 to t

            if len(ret_aligned) == len(vol_changes):
                rho = float(np.corrcoef(ret_aligned, vol_changes)[0, 1])
                rho = np.clip(rho, -0.9, -0.3)  # Typically negative for equities
            else:
                rho = -0.7  # Default for equities
        else:
            rho = -0.7  # Default

        print(f"✓ Estimated Heston parameters:")
        print(f"  v0 (initial var):  {v0:.6f}  (vol={np.sqrt(v0):.2%}) [from recent 3M]")
        print(f"  θ (long-term var): {theta:.6f}  (vol={np.sqrt(theta):.2%}) [from full sample]")
        print(f"  κ (mean reversion):{kappa:.4f} [from full sample]")
        print(f"  σ (vol of vol):    {sigma:.4f} [from full sample]")
        print(f"  ρ (correlation):   {rho:.4f} [from full sample]")

        return cls(
            v0=v0,
            theta=theta,
            kappa=kappa,
            sigma=sigma,
            rho=rho,
            calibrated=True,
            calibration_instruments=[f"Historical prices: {total_days} days (v0 from recent 63 days)"]
        )

    def validate(self) -> bool:
        """Validate Feller condition: 2*kappa*theta > sigma^2"""
        feller = 2 * self.kappa * self.theta > self.sigma ** 2
        if not feller:
            print(f"⚠ Warning: Feller condition violated. 2κθ={2*self.kappa*self.theta:.4f} < σ²={self.sigma**2:.4f}")
        else:
            print(f"✓ Feller condition satisfied: 2κθ={2*self.kappa*self.theta:.4f} > σ²={self.sigma**2:.4f}")
        return feller


# =============================================================================
# 3. HULL-WHITE MODEL CALIBRATION
# =============================================================================

@dataclass
class HullWhiteModelParams:
    """
    Hull-White (1-factor) model parameters for interest rate simulation.

    dr_t = (theta_t - a*r_t)dt + sigma dW_t

    Also known as extended Vasicek or G1++ model in QuantLib.
    """
    # Mean reversion speed
    mean_reversion: float  # 'a' parameter - speed of mean reversion

    # Volatility
    sigma: float  # Short rate volatility

    # Calibration instruments (e.g., swaptions for vol calibration)
    calibration_instruments: Optional[List] = None
    calibrated: bool = False

    @classmethod
    def calibrate_from_rates(cls, rates_data: RatesMarketData,
                            mean_reversion: float = 0.05,
                            sigma: float = 0.01):
        """
        Simple Hull-White calibration using typical market parameters.

        For a more sophisticated calibration, you would calibrate to swaption prices.
        This simplified version uses reasonable default parameters for USD rates.

        Args:
            rates_data: RatesMarketData with yield curve
            mean_reversion: Mean reversion speed (default 0.05 = 5% for USD)
            sigma: Short rate volatility (default 0.01 = 1% for USD)

        Returns:
            HullWhiteModelParams with calibrated parameters
        """
        print(f"✓ Hull-White Model Parameters:")
        print(f"  Mean reversion (a): {mean_reversion:.4f}")
        print(f"  Volatility (σ):     {sigma:.4f}")
        print(f"  Note: Using typical USD parameters. For production, calibrate to swaptions.")

        return cls(
            mean_reversion=mean_reversion,
            sigma=sigma,
            calibrated=True,
            calibration_instruments=[f"Yield curve with {len(rates_data.tenor_points)} points"]
        )


# =============================================================================
# 4. MAIN CALIBRATION ROUTINE
# =============================================================================

def calibrate_models_from_files(option_data_file: str,
                                spot_price: float,
                                rates_data: Optional[RatesMarketData] = None,
                                historical_prices_file: Optional[str] = None):
    """
    Main calibration routine using LOCAL FILES instead of live market data.

    This is useful when you cannot access Yahoo Finance or other APIs.

    Args:
        option_data_file: Path to Excel file with option data (Strike, Implied Volatility, etc.)
        spot_price: Current spot price of the underlying
        rates_data: Pre-constructed RatesMarketData (or None to use defaults)
        historical_prices_file: Optional path to historical prices file for better calibration

    Returns:
        Dictionary with calibrated parameters and market data

    Example:
        calibrated = calibrate_models_from_files(
            option_data_file="Exotic/spx_infvol_20260109.xlsx",
            spot_price=5900.0,
            rates_data=None  # Will use default rates
        )
    """
    print("="*80)
    print("MODEL CALIBRATION - HESTON & HULL-WHITE (FILE-BASED)")
    print("="*80)
    print(f"📁 Using local files instead of live market data")
    print()

    # 1. Load equity market data from Excel
    print("STEP 1: Loading Equity Market Data from File")
    print("-" * 80)
    equity_data = EquityMarketData.from_excel(
        excel_file=option_data_file,
        spot_price=spot_price,
        historical_prices_file=historical_prices_file,
        repo_rate=0.0
    )
    print()

    # 2. Use rates market data (or create default)
    print("STEP 2: Setting Up Rates Market Data")
    print("-" * 80)
    if rates_data is None:
        print("  Using default USD rate curve (manually specified)")
        # Default USD curve (example values - update with current market data)
        rates_data = RatesMarketData(
            curve_date=datetime.now(),
            tenor_points=["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"],
            zero_rates=[0.045, 0.046, 0.045, 0.043, 0.041, 0.039, 0.038],
            day_count="ACT/360",
            calendar="UnitedStates"
        )
        for tenor, rate in zip(rates_data.tenor_points, rates_data.zero_rates):
            print(f"  {tenor:>4s}: {rate:>6.2%}")
    else:
        print("  Using provided rates data")
    print()

    # 3. Calibrate Heston model
    print("STEP 3: Calibrating Heston Model")
    print("-" * 80)

    # Choose calibration method based on available data
    if equity_data.historical_prices is not None and len(equity_data.historical_prices) >= 252:
        print("  Using historical prices for calibration (recommended)")
        heston_params = HestonModelParams.estimate_from_historical(equity_data)
    else:
        print("  Using volatility surface for calibration")
        heston_params = HestonModelParams.estimate_from_vol_surface(equity_data)
    print()

    # 4. Validate Heston parameters
    print("STEP 4: Validating Heston Parameters")
    print("-" * 80)
    heston_params.validate()
    print()

    # 5. Calibrate Hull-White model
    print("STEP 5: Calibrating Hull-White Model")
    print("-" * 80)
    hw_params = HullWhiteModelParams.calibrate_from_rates(rates_data)
    print()

    # 6. Summary
    print("="*80)
    print("CALIBRATION COMPLETE ✓")
    print("="*80)
    print()
    print("Calibration Summary:")
    print(f"  Equity Data Source:  {option_data_file}")
    print(f"  Spot Price:          {spot_price:,.2f}")
    print(f"  ATM Volatility:      {equity_data.atm_volatility:.2%}")
    print(f"  Vol Surface Points:  {len(equity_data.vol_surface) if equity_data.vol_surface else 0}")
    print()
    print("You can now use these calibrated parameters in your simulation:")
    print()
    print("  from model_calibration import calibrate_models_from_files")
    print("  calibrated = calibrate_models_from_files(...)")
    print("  heston_params = calibrated['heston_params']")
    print("  hw_params = calibrated['hull_white_params']")
    print()
    print("="*80)

    return {
        'equity_data': equity_data,
        'rates_data': rates_data,
        'heston_params': heston_params,
        'hull_white_params': hw_params
    }


def calibrate_models(equity_ticker: str = "^GSPC",
                     lookback_days: int = 756,
                     rate_limit_delay: float = 2.0):
    """
    Main calibration routine for Heston and Hull-White models.

    Uses Yahoo Finance for equity data with rate limiting and retry logic.

    DEPRECATED: If Yahoo Finance is not accessible, use calibrate_models_from_files() instead.

    Args:
        equity_ticker: Equity ticker for Heston calibration (default "^GSPC" for SPX)
        lookback_days: Days of historical data (default 756 = 3 years)
        rate_limit_delay: Minimum seconds between API requests (default 2.0)

    Returns:
        Dictionary with calibrated parameters and market data
    """
    # Set rate limiter delay
    global _rate_limiter
    _rate_limiter = RateLimiter(min_interval=rate_limit_delay)

    print("="*80)
    print("MODEL CALIBRATION - HESTON & HULL-WHITE")
    print("="*80)
    print(f"⏱ Rate limiting enabled: {rate_limit_delay}s minimum between requests")
    print(f"🔄 Retry logic enabled: up to 3 attempts with exponential backoff")
    print()

    # 1. Fetch equity market data
    print("STEP 1: Fetching Equity Market Data (Yahoo Finance)")
    print("-" * 80)
    equity_data = EquityMarketData.from_yahoo(
        ticker=equity_ticker,
        lookback_days=lookback_days,
        repo_rate=0.0
    )
    print()

    # 2. Fetch rates market data
    print("STEP 2: Fetching Rates Market Data (FRED)")
    print("-" * 80)
    rates_data = RatesMarketData.from_fred(currency="USD")
    print()

    # 3. Calibrate Heston model
    print("STEP 3: Calibrating Heston Model")
    print("-" * 80)
    heston_params = HestonModelParams.estimate_from_historical(equity_data)
    print()

    # 4. Validate Heston parameters
    print("STEP 4: Validating Heston Parameters")
    print("-" * 80)
    heston_params.validate()
    print()

    # 5. Calibrate Hull-White model
    print("STEP 5: Calibrating Hull-White Model")
    print("-" * 80)
    hw_params = HullWhiteModelParams.calibrate_from_rates(rates_data)
    print()

    # 6. Summary
    print("="*80)
    print("CALIBRATION COMPLETE ✓")
    print("="*80)
    print()
    print("You can now import these calibrated parameters in your main simulation:")
    print()
    print("  from model_calibration import calibrate_models")
    print("  calibrated = calibrate_models()")
    print("  heston_params = calibrated['heston_params']")
    print("  hw_params = calibrated['hull_white_params']")
    print()
    print("="*80)

    return {
        'equity_data': equity_data,
        'rates_data': rates_data,
        'heston_params': heston_params,
        'hull_white_params': hw_params
    }


# =============================================================================
# MAIN - RUN CALIBRATION
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check if user wants to use file-based calibration
    use_files = len(sys.argv) > 1 and sys.argv[1] == "--files"

    if use_files:
        print("Using FILE-BASED calibration (offline mode)")
        print()

        # Example: Calibrate using local Excel file
        calibrated = calibrate_models_from_files(
            option_data_file="Exotic/spx_infvol_20260109.xlsx",  # Path to your Excel file
            spot_price=5900.0,  # Current SPX level (provide manually)
            rates_data=None,  # Will use default rates
            historical_prices_file=None  # Optional: provide historical prices CSV/Excel
        )

    else:
        print("Using LIVE DATA calibration (online mode)")
        print("Tip: Use 'python model_calibration.py --files' for offline calibration")
        print()

        # Example: Calibrate models using Yahoo Finance data for SPX
        try:
            calibrated = calibrate_models(
                equity_ticker="^GSPC",  # S&P 500 index
                lookback_days=756,  # 3 years of data
                rate_limit_delay=2.0  # 2 seconds between requests
            )
        except Exception as e:
            print(f"\n✗ Error fetching live data: {e}")
            print("\nTry using file-based calibration instead:")
            print("  python model_calibration.py --files")
            sys.exit(1)

    # Access calibrated parameters
    heston = calibrated['heston_params']
    hull_white = calibrated['hull_white_params']

    print("\n✓ Calibrated parameters are ready to use!")
    print("\nNext steps:")
    print("  1. Import these parameters in Barrier_option_inputs.py")
    print("  2. Use them for PFE simulation and barrier option pricing")
