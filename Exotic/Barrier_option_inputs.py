"""
Modular Input Framework for Exotic Equity Option PFE Simulation
=================================================================

This module defines the inputs needed for simulating Potential Future Exposure (PFE)
for exotic equity options including barrier options, considering both equity and rates risk.

Target: 10-day Margin Period of Risk (MPOR), 99% quantile PFE
"""

# QuantLib is optional - only required for QuantLib conversion methods
try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False
    print("Warning: QuantLib not installed. QuantLib conversion methods will not be available.")
    print("Install with: pip install QuantLib-Python")

from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np


# =============================================================================
# 1. PRODUCT PARAMETERS MODULE
# =============================================================================

@dataclass
class BarrierOptionParams:
    """
    Product-specific parameters for barrier options.
    Modular design allows easy extension to other exotic options.
    """
    # Basic option parameters
    option_type: str  # "Put" or "Call"
    strike: float  # Option strike price
    notional: float  # Number of units
    multiplier: float  # Contract multiplier

    # Barrier-specific parameters
    barrier_type: str  # "DownOut", "DownIn", "UpOut", "UpIn"
    barrier_level: float  # Knock-out/in barrier level

    # Temporal parameters
    valuation_date: datetime  # Trade date / valuation date
    expiry_date: datetime  # Option expiration date

    # Underlying
    underlying_ticker: str  # e.g., "SPX"
    currency: str  # e.g., "USD"

    @property
    def tenor_months(self) -> int:
        """Calculate tenor in months from valuation_date to expiry_date"""
        return (self.expiry_date.year - self.valuation_date.year) * 12 + \
               (self.expiry_date.month - self.valuation_date.month)

    def to_quantlib(self, calculation_date) -> Dict:
        """Convert to QuantLib objects"""
        if not HAS_QUANTLIB:
            raise ImportError("QuantLib is required for this method. Install with: pip install QuantLib-Python")

        # Convert option type
        option_type_ql = ql.Option.Put if self.option_type.lower() == "put" else ql.Option.Call

        # Convert barrier type
        barrier_map = {
            "downout": ql.Barrier.DownOut,
            "downin": ql.Barrier.DownIn,
            "upout": ql.Barrier.UpOut,
            "upin": ql.Barrier.UpIn
        }
        barrier_type_ql = barrier_map[self.barrier_type.lower()]

        # Convert dates
        expiry_ql = ql.Date(self.expiry_date.day, self.expiry_date.month, self.expiry_date.year)

        return {
            'option_type': option_type_ql,
            'barrier_type': barrier_type_ql,
            'strike': self.strike,
            'barrier': self.barrier_level,
            'expiry_date': expiry_ql,
            'notional': self.notional,
            'multiplier': self.multiplier
        }


# =============================================================================
# 2. MARKET DATA INPUT MODULE
# =============================================================================

@dataclass
class EquityMarketData:
    """
    Market data for equity underlying (SPX in this case).
    """
    # Spot data
    spot_price: float  # Current SPX level

    # Volatility surface data (for calibration)
    # Simplified: can be expanded to full surface
    atm_volatility: float  # ATM implied volatility (historical vol)
    vol_surface: Optional[Dict[tuple, float]] = None  # (strike, tenor) -> vol

    # Historical data for calibration
    historical_prices: Optional[List[float]] = None  # Time series for calibration

    # Repo rate / borrow cost
    repo_rate: float = 0.0  # Equity repo rate (usually small)

    # Dividend yield (ignored per requirement, but included for completeness)
    dividend_yield: float = 0.0

    @classmethod
    def from_alpha_vantage(cls, ticker: str, api_key: str, lookback_days: int = 756,
                          repo_rate: float = 0.0, outputsize: str = 'full'):
        """
        Fetch equity market data from Alpha Vantage API.

        Args:
            ticker: Stock ticker (e.g., "SPY" for SPY ETF, "AAPL" for Apple)
                   Note: Use "SPY" instead of "^GSPC" for S&P 500
            api_key: Alpha Vantage API key
            lookback_days: Days of historical data for volatility calculation
                          (default 756 = 3 years, recommended for Heston calibration)
            repo_rate: Equity repo rate (default 0.0, usually small for indices)
            outputsize: 'compact' (100 days) or 'full' (20+ years), default 'full'

        Returns:
            EquityMarketData instance with current spot and historical volatility

        Note:
            Alpha Vantage has rate limits:
            - Free tier: 25 requests/day, 5 requests/minute
            - Premium tier: Higher limits available
        """
        try:
            import requests
            import pandas as pd
        except ImportError:
            raise ImportError("Please install: pip install requests pandas")

        print(f"✓ Fetching data from Alpha Vantage for {ticker}...")

        # Alpha Vantage TIME_SERIES_DAILY endpoint (free tier)
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',  # Changed from DAILY_ADJUSTED (premium)
            'symbol': ticker,
            'apikey': api_key,
            'outputsize': outputsize,
            'datatype': 'json'
        }

        response = requests.get(url, params=params)
        data = response.json()

        # Check for errors
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        if 'Note' in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
        if 'Information' in data:
            raise ValueError(f"Alpha Vantage API message: {data['Information']}")
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"Unexpected response format. Keys: {data.keys()}\nFull response: {data}")

        # Parse time series data
        time_series = data['Time Series (Daily)']

        # Convert to pandas DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()  # Sort chronologically

        # Alpha Vantage columns for TIME_SERIES_DAILY: '1. open', '2. high', '3. low', '4. close', '5. volume'
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Limit to lookback_days
        if len(df) > lookback_days:
            df = df.iloc[-lookback_days:]

        # Get spot price (most recent close)
        # Note: TIME_SERIES_DAILY doesn't have adjusted close, only regular close
        spot_price = float(df['Close'].iloc[-1])

        # Calculate historical volatility (annualized) using close
        returns = df['Close'].pct_change().dropna()
        historical_vol = float(returns.std() * np.sqrt(252))  # Annualized volatility

        # Get historical prices as list (close)
        historical_prices = df['Close'].tolist()

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
    def from_yahoo(cls, ticker: str, lookback_days: int = 756, repo_rate: float = 0.0):
        """
        Fetch equity market data from Yahoo Finance (free).

        DEPRECATED: Consider using from_alpha_vantage() instead.

        Args:
            ticker: Stock ticker (e.g., "^GSPC" for SPX, "AAPL" for Apple)
            lookback_days: Days of historical data for volatility calculation
                          (default 756 = 3 years, recommended for Heston calibration)
            repo_rate: Equity repo rate (default 0.0, usually small for indices)

        Returns:
            EquityMarketData instance with current spot and historical volatility
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Please install yfinance: pip install yfinance")

        # Download data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=f"{lookback_days}d")

        if hist_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        # Get spot price (most recent close)
        spot_price = float(hist_data['Close'].iloc[-1])

        # Calculate historical volatility (annualized)
        returns = hist_data['Close'].pct_change().dropna()
        historical_vol = float(returns.std() * np.sqrt(252))  # Annualized volatility

        # Get historical prices as list
        historical_prices = hist_data['Close'].tolist()

        print(f"✓ Fetched data for {ticker}:")
        print(f"  Spot Price: {spot_price:,.2f}")
        print(f"  Historical Vol (1Y): {historical_vol:.2%}")
        print(f"  Historical Prices: {len(historical_prices)} days")

        return cls(
            spot_price=spot_price,
            atm_volatility=historical_vol,
            vol_surface=None,  # Can be extended for term structure of vol
            historical_prices=historical_prices,
            repo_rate=repo_rate,
            dividend_yield=0.0  # Ignored per requirement
        )

    @classmethod
    def from_yahoo_with_options(cls, ticker: str, lookback_days: int = 756,
                                repo_rate: float = 0.0, risk_free_rate: float = 0.045):
        """
        Fetch equity market data AND build volatility surface from option prices.

        This method fetches actual option market prices and calculates implied volatilities
        to build a proper volatility surface for Heston calibration.

        Args:
            ticker: Stock ticker (e.g., "SPY" for SPY ETF, note: "^GSPC" doesn't have options)
            lookback_days: Days of historical data (default 756 = 3 years)
            repo_rate: Equity repo rate (default 0.0)
            risk_free_rate: Risk-free rate for Black-Scholes IV calculation (default 4.5%)

        Returns:
            EquityMarketData instance with spot, historical data, AND volatility surface

        Note:
            - For SPX exposure, use "SPY" (SPX ETF) as ^GSPC doesn't have listed options
            - Requires scipy for implied volatility calculation
        """
        try:
            import yfinance as yf
            from scipy.optimize import brentq
            from scipy.stats import norm
        except ImportError:
            raise ImportError("Please install: pip install yfinance scipy")

        print(f"✓ Fetching market data and options for {ticker}...")

        # 1. Get underlying price and historical data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=f"{lookback_days}d")

        if hist_data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        spot_price = float(hist_data['Close'].iloc[-1])

        # Calculate historical volatility
        returns = hist_data['Close'].pct_change().dropna()
        historical_vol = float(returns.std() * np.sqrt(252))
        historical_prices = hist_data['Close'].tolist()

        print(f"  Spot Price: {spot_price:,.2f}")
        print(f"  Historical Vol: {historical_vol:.2%}")

        # 2. Get option expirations
        try:
            expirations = stock.options
            if not expirations:
                print("  Warning: No options data available, using historical vol only")
                return cls(
                    spot_price=spot_price,
                    atm_volatility=historical_vol,
                    vol_surface=None,
                    historical_prices=historical_prices,
                    repo_rate=repo_rate,
                    dividend_yield=0.0
                )
        except Exception as e:
            print(f"  Warning: Could not fetch options data: {e}")
            return cls(
                spot_price=spot_price,
                atm_volatility=historical_vol,
                vol_surface=None,
                historical_prices=historical_prices,
                repo_rate=repo_rate,
                dividend_yield=0.0
            )

        # 3. Build volatility surface from options
        print(f"  Found {len(expirations)} option expirations")

        vol_surface = {}
        atm_vols = []

        from datetime import datetime
        today = datetime.now()

        # Process first few expirations (to avoid too much data)
        for expiry_str in expirations[:6]:  # Use first 6 maturities
            try:
                opt_chain = stock.option_chain(expiry_str)
                puts = opt_chain.puts

                # Calculate time to expiry in years
                expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d')
                days_to_expiry = (expiry_date - today).days
                if days_to_expiry <= 0:
                    continue

                T = days_to_expiry / 365.0
                tenor = f"{days_to_expiry}D"

                # Filter for liquid options (volume > 0, bid > 0)
                puts = puts[(puts['volume'] > 0) & (puts['bid'] > 0)]

                if puts.empty:
                    continue

                # Calculate implied vol for each strike
                for _, row in puts.iterrows():
                    strike = float(row['strike'])

                    # Use mid price
                    option_price = (float(row['bid']) + float(row['ask'])) / 2.0

                    if option_price <= 0.01:  # Skip very cheap options
                        continue

                    # Calculate implied volatility using Black-Scholes
                    try:
                        iv = cls._calculate_implied_vol(
                            option_price, spot_price, strike, T,
                            risk_free_rate, option_type='put'
                        )

                        if iv is not None and 0.05 < iv < 2.0:  # Sanity check
                            vol_surface[(strike, tenor)] = iv

                            # Track ATM vol (strike closest to spot)
                            if abs(strike - spot_price) / spot_price < 0.05:  # Within 5%
                                atm_vols.append(iv)
                    except:
                        continue

            except Exception as e:
                print(f"  Warning: Could not process expiry {expiry_str}: {e}")
                continue

        # Calculate ATM implied vol (average of near-ATM options)
        if atm_vols:
            atm_implied_vol = float(np.median(atm_vols))
            print(f"  Implied ATM Vol: {atm_implied_vol:.2%} (from {len(atm_vols)} near-ATM options)")
        else:
            atm_implied_vol = historical_vol
            print(f"  Using Historical Vol as ATM: {atm_implied_vol:.2%}")

        if vol_surface:
            print(f"  Built volatility surface: {len(vol_surface)} points")

            # Print sample of vol surface
            print(f"  Sample volatility surface:")
            sample_points = list(vol_surface.items())[:5]
            for (strike, tenor), vol in sample_points:
                moneyness = strike / spot_price
                print(f"    Strike {strike:.0f} ({moneyness:.2%} moneyness), {tenor}: {vol:.2%}")
        else:
            print(f"  Warning: Could not build vol surface, using historical vol")
            vol_surface = None

        return cls(
            spot_price=spot_price,
            atm_volatility=atm_implied_vol,
            vol_surface=vol_surface,
            historical_prices=historical_prices,
            repo_rate=repo_rate,
            dividend_yield=0.0
        )

    @staticmethod
    def _calculate_implied_vol(option_price, spot, strike, T, r, option_type='put',
                               max_iterations=100, tolerance=1e-6):
        """
        Calculate implied volatility using Black-Scholes and Newton-Raphson method.

        Args:
            option_price: Market price of the option
            spot: Current underlying price
            strike: Option strike price
            T: Time to expiry in years
            r: Risk-free rate
            option_type: 'put' or 'call'

        Returns:
            Implied volatility (annualized) or None if calculation fails
        """
        try:
            from scipy.stats import norm
            from scipy.optimize import brentq
        except ImportError:
            return None

        def black_scholes_price(S, K, T, r, sigma, option_type='put'):
            """Black-Scholes option pricing formula"""
            if T <= 0 or sigma <= 0:
                return 0

            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return price

        def objective(sigma):
            """Objective function: model_price - market_price"""
            return black_scholes_price(spot, strike, T, r, sigma, option_type) - option_price

        try:
            # Use Brent's method to find root (implied vol)
            # Search between 1% and 300% volatility
            implied_vol = brentq(objective, 0.01, 3.0, maxiter=max_iterations, xtol=tolerance)
            return implied_vol
        except:
            return None

    def to_quantlib(self, calculation_date) -> Dict:
        """Convert to QuantLib market data objects"""
        if not HAS_QUANTLIB:
            raise ImportError("QuantLib is required for this method. Install with: pip install QuantLib-Python")

        # Spot handle
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.spot_price))

        # Dividend yield (set to 0 as per requirement)
        dividend_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(calculation_date, self.dividend_yield, ql.Actual365Fixed())
        )

        # Simple flat volatility (will be replaced by Heston)
        volatility_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(calculation_date, ql.NullCalendar(),
                               self.atm_volatility, ql.Actual365Fixed())
        )

        return {
            'spot_handle': spot_handle,
            'dividend_ts': dividend_ts,
            'volatility_ts': volatility_ts
        }


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

        print(f"✓ Fetching most recent USD rates from FRED (last 30 days)")

        for tenor, series_code in fred_series.items():
            try:
                # Fetch data from FRED - get last 30 days to ensure we get data
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
            curve_date=date,
            tenor_points=tenor_points,
            zero_rates=zero_rates,
            day_count="ACT/360",
            calendar="UnitedStates"
        )

    def to_quantlib(self, calculation_date):
        """Convert to QuantLib yield curve"""
        if not HAS_QUANTLIB:
            raise ImportError("QuantLib is required for this method. Install with: pip install QuantLib-Python")

        # Parse tenors to QuantLib dates
        dates = [calculation_date]
        rates = []

        for tenor_str, rate in zip(self.tenor_points, self.zero_rates):
            # Parse tenor string (simplified parser)
            if 'D' in tenor_str:
                days = int(tenor_str.replace('D', ''))
                dates.append(calculation_date + ql.Period(days, ql.Days))
            elif 'W' in tenor_str:
                weeks = int(tenor_str.replace('W', ''))
                dates.append(calculation_date + ql.Period(weeks, ql.Weeks))
            elif 'M' in tenor_str:
                months = int(tenor_str.replace('M', ''))
                dates.append(calculation_date + ql.Period(months, ql.Months))
            elif 'Y' in tenor_str:
                years = int(tenor_str.replace('Y', ''))
                dates.append(calculation_date + ql.Period(years, ql.Years))

            rates.append(rate)

        # Create yield curve
        day_count_ql = ql.Actual360() if self.day_count == "ACT/360" else ql.Actual365Fixed()

        curve = ql.ZeroCurve(dates[1:], rates, day_count_ql)

        return ql.YieldTermStructureHandle(curve)


# =============================================================================
# 3. HESTON MODEL PARAMETERS MODULE
# =============================================================================

@dataclass
class HestonModelParams:
    """
    Heston stochastic volatility model parameters for equity simulation.

    dS_t = r S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa(theta - v_t)dt + sigma sqrt(v_t) dW^v_t
    dW^S_t dW^v_t = rho dt

    NOTE: For calibration, use the model_calibration.py module separately.
          This module focuses on using pre-calibrated parameters.
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

    def validate(self) -> bool:
        """Validate Feller condition: 2*kappa*theta > sigma^2"""
        feller = 2 * self.kappa * self.theta > self.sigma ** 2
        if not feller:
            print(f"Warning: Feller condition violated. 2κθ={2*self.kappa*self.theta:.4f} < σ²={self.sigma**2:.4f}")
        return feller

    def to_quantlib(self):
        """
        Create QuantLib Heston model (requires process setup with curves).
        This returns parameters; actual model creation happens in simulation module.
        """
        return {
            'v0': self.v0,
            'theta': self.theta,
            'kappa': self.kappa,
            'sigma': self.sigma,
            'rho': self.rho
        }


# =============================================================================
# 4. HULL-WHITE (G1++) MODEL PARAMETERS MODULE
# =============================================================================

@dataclass
class HullWhiteModelParams:
    """
    Hull-White (1-factor) model parameters for interest rate simulation.

    dr_t = (theta_t - a*r_t)dt + sigma dW_t

    Also known as extended Vasicek or G1++ model in QuantLib.

    NOTE: For calibration, use the model_calibration.py module separately.
          This module focuses on using pre-calibrated parameters.
    """
    # Mean reversion speed
    mean_reversion: float  # 'a' parameter - speed of mean reversion

    # Volatility
    sigma: float  # Short rate volatility

    # Time-dependent drift (fitted to yield curve)
    # theta_t is calibrated from initial yield curve

    # Calibration instruments (e.g., swaptions for vol calibration)
    calibration_instruments: Optional[List] = None
    calibrated: bool = False

    def to_quantlib(self, yield_curve):
        """
        Create QuantLib Hull-White model.
        """
        # Create Hull-White model with fitted term structure
        hw_model = ql.HullWhite(yield_curve, self.mean_reversion, self.sigma)

        return hw_model


# =============================================================================
# 5. SIMULATION PARAMETERS MODULE
# =============================================================================

@dataclass
class SimulationParams:
    """
    Monte Carlo simulation configuration for PFE calculation.
    """
    # PFE-specific parameters
    mpor_days: int  # Margin Period of Risk (10 days for this case)
    confidence_level: float  # 99% for 99th percentile

    # Monte Carlo parameters
    num_paths: int  # Number of simulation paths (e.g., 10,000 - 100,000)
    num_time_steps: int  # Time steps per MPOR period

    # Random number generation
    seed: int  # For reproducibility
    antithetic: bool = True  # Use antithetic variates for variance reduction

    # Simulation grid
    # For PFE: need to simulate from t=0 to t=MPOR
    simulation_dates: Optional[List[datetime]] = None

    def get_time_grid(self, start_date, end_date):
        """Create QuantLib time grid for simulation"""
        time_grid = ql.TimeGrid(
            (end_date - start_date) / 365.0,  # Total time in years
            self.num_time_steps
        )
        return time_grid


# =============================================================================
# 6. EXAMPLE USAGE - SPX DOWN-OUT PUT
# =============================================================================

def create_spx_barrier_example(use_calibrated_models: bool = False):
    """
    Create input configuration for the SPX barrier option example.

    Product: Down-Out Put on SPX
    - Strike: 6500
    - Barrier: 6000 (knock-out)
    - Tenor: 9 months
    - Notional: 100 units x 1 multiplier
    - MPOR: 10 days
    - Confidence: 99%

    Args:
        use_calibrated_models: If True, import calibrated parameters from model_calibration.py
                              If False, use hardcoded example parameters (default)
    """

    # 1. Product Parameters
    product = BarrierOptionParams(
        option_type="Put",
        strike=6500.0,
        barrier_type="DownOut",
        barrier_level=6000.0,
        notional=100.0,
        multiplier=1.0,
        valuation_date=datetime(2026, 1, 9),  # Today
        expiry_date=datetime(2026, 10, 9),  # 9 months later
        underlying_ticker="SPX",
        currency="USD"
    )

    # 2. Market Data and Model Calibration
    if use_calibrated_models:
        print("\n✓ Using calibrated models from model_calibration.py")
        print("=" * 80)

        # Import calibration module
        try:
            from model_calibration import calibrate_models

            # Run calibration (fetches live market data)
            calibrated = calibrate_models(
                equity_ticker="^GSPC",  # S&P 500 index
                use_yahoo=True,
                lookback_days=756  # 3 years of data
            )

            # Extract calibrated data
            equity_data = calibrated['equity_data']
            rates_data = calibrated['rates_data']
            heston_params = calibrated['heston_params']
            hull_white_params = calibrated['hull_white_params']

        except ImportError as e:
            print(f"\n⚠ Warning: Could not import model_calibration: {e}")
            print("  Falling back to hardcoded parameters...")
            use_calibrated_models = False

    if not use_calibrated_models:
        print("\n✓ Using hardcoded example parameters")
        print("=" * 80)

        # 2. Equity Market Data (example values - replace with actual market data)
        equity_data = EquityMarketData(
            spot_price=5900.0,  # Current SPX level (example)
            atm_volatility=0.15,  # 15% ATM vol (example)
            repo_rate=0.045,  # 4.5% repo rate (example)
            dividend_yield=0.0,  # Ignored per requirement
            vol_surface=None,  # Can add full surface for calibration
            historical_prices=None  # Can add for historical calibration
        )

        # 3. Rates Market Data (example USD curve - replace with actual market data)
        rates_data = RatesMarketData(
            curve_date=datetime(2026, 1, 9),
            tenor_points=["1D", "1W", "1M", "3M", "6M", "9M", "1Y", "2Y", "5Y", "10Y"],
            zero_rates=[
                0.0450,  # 1D: 4.50%
                0.0455,  # 1W: 4.55%
                0.0460,  # 1M: 4.60%
                0.0465,  # 3M: 4.65%
                0.0450,  # 6M: 4.50%
                0.0440,  # 9M: 4.40%
                0.0430,  # 1Y: 4.30%
                0.0410,  # 2Y: 4.10%
                0.0390,  # 5Y: 3.90%
                0.0380   # 10Y: 3.80%
            ],
            day_count="ACT/360",
            calendar="UnitedStates"
        )

        # 4. Heston Model Parameters (example - should be calibrated to market)
        heston_params = HestonModelParams(
            v0=0.0225,      # Initial variance (15% vol squared)
            theta=0.0225,   # Long-term variance
            kappa=2.0,      # Mean reversion speed
            sigma=0.3,      # Vol of vol (30%)
            rho=-0.7,       # Negative correlation (typical for equity)
            calibrated=False
        )

        # Validate Feller condition
        heston_params.validate()

        # 5. Hull-White Model Parameters (example - should be calibrated)
        hull_white_params = HullWhiteModelParams(
            mean_reversion=0.05,  # 5% mean reversion (typical for USD)
            sigma=0.01,           # 1% short rate volatility
            calibrated=False
        )

    # 6. Simulation Parameters
    sim_params = SimulationParams(
        mpor_days=10,              # 10-day margin period of risk
        confidence_level=0.99,     # 99% quantile
        num_paths=50000,           # 50,000 paths for smooth quantile
        num_time_steps=40,         # 4 steps per day (10 days * 4)
        seed=42,                   # Reproducibility
        antithetic=True            # Variance reduction
    )

    return {
        'product': product,
        'equity_data': equity_data,
        'rates_data': rates_data,
        'heston_params': heston_params,
        'hull_white_params': hull_white_params,
        'simulation_params': sim_params
    }


# =============================================================================
# 7. INPUT SUMMARY AND VALIDATION
# =============================================================================

def print_input_summary(inputs: Dict):
    """Print summary of all inputs for review"""

    print("="*80)
    print("BARRIER OPTION PFE SIMULATION - INPUT SUMMARY")
    print("="*80)

    print("\n1. PRODUCT PARAMETERS")
    print("-" * 80)
    prod = inputs['product']
    print(f"   Underlying:        {prod.underlying_ticker}")
    print(f"   Option Type:       {prod.barrier_type} {prod.option_type}")
    print(f"   Strike:            {prod.strike:,.2f}")
    print(f"   Barrier Level:     {prod.barrier_level:,.2f}")
    print(f"   Notional:          {prod.notional:.0f} units x {prod.multiplier}x multiplier")
    print(f"   Valuation Date:    {prod.valuation_date.strftime('%Y-%m-%d')}")
    print(f"   Expiry Date:       {prod.expiry_date.strftime('%Y-%m-%d')}")
    print(f"   Tenor:             {prod.tenor_months} months")

    print("\n2. EQUITY MARKET DATA")
    print("-" * 80)
    eq = inputs['equity_data']
    print(f"   Spot Price:        {eq.spot_price:,.2f}")
    print(f"   ATM Volatility:    {eq.atm_volatility:.2%}")
    print(f"   Repo Rate:         {eq.repo_rate:.2%}")
    print(f"   Dividend Yield:    {eq.dividend_yield:.2%} (ignored)")

    print("\n3. RATES MARKET DATA")
    print("-" * 80)
    rates = inputs['rates_data']
    print(f"   Curve Date:        {rates.curve_date.strftime('%Y-%m-%d')}")
    print(f"   Day Count:         {rates.day_count}")
    print(f"   Calendar:          {rates.calendar}")
    print(f"   Curve Points:")
    for tenor, rate in zip(rates.tenor_points, rates.zero_rates):
        print(f"      {tenor:>4s}: {rate:>6.2%}")

    print("\n4. HESTON MODEL PARAMETERS")
    print("-" * 80)
    hest = inputs['heston_params']
    print(f"   v0 (initial var):  {hest.v0:.6f}  (vol={np.sqrt(hest.v0):.2%})")
    print(f"   θ (long-term var): {hest.theta:.6f}  (vol={np.sqrt(hest.theta):.2%})")
    print(f"   κ (mean reversion):{hest.kappa:.4f}")
    print(f"   σ (vol of vol):    {hest.sigma:.4f}")
    print(f"   ρ (correlation):   {hest.rho:.4f}")
    print(f"   Feller condition:  {'OK' if hest.validate() else 'VIOLATED'}")
    print(f"   Calibrated:        {hest.calibrated}")

    print("\n5. HULL-WHITE MODEL PARAMETERS")
    print("-" * 80)
    hw = inputs['hull_white_params']
    print(f"   a (mean reversion):{hw.mean_reversion:.4f}")
    print(f"   σ (volatility):    {hw.sigma:.4f}")
    print(f"   Calibrated:        {hw.calibrated}")

    print("\n6. SIMULATION PARAMETERS")
    print("-" * 80)
    sim = inputs['simulation_params']
    print(f"   MPOR:              {sim.mpor_days} days")
    print(f"   Confidence Level:  {sim.confidence_level:.1%}")
    print(f"   Number of Paths:   {sim.num_paths:,}")
    print(f"   Time Steps:        {sim.num_time_steps}")
    print(f"   Seed:              {sim.seed}")
    print(f"   Antithetic:        {sim.antithetic}")

    print("\n" + "="*80)
    print("INPUTS READY FOR SIMULATION MODULE")
    print("="*80 + "\n")


# =============================================================================
# MAIN - EXAMPLE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Create example inputs for SPX barrier option
    inputs = create_spx_barrier_example()

    # Print comprehensive summary
    print_input_summary(inputs)

    # Validate Heston parameters
    print("\nValidating Heston parameters...")
    inputs['heston_params'].validate()

    print("\n✓ All input modules initialized successfully!")
    print("\nNext steps:")
    print("  1. Implement Heston simulation module")
    print("  2. Implement Hull-White simulation module")
    print("  3. Implement barrier option pricing module")
    print("  4. Implement PFE calculation module")
    print("  5. Run full simulation and calculate 99% 10-day MPOR PFE")
