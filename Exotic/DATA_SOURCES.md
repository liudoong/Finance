# Data Source Comparison: Alpha Vantage vs Yahoo Finance

## Summary

| Feature | Alpha Vantage | Yahoo Finance |
|---------|---------------|---------------|
| **Historical Prices** | ✅ Yes (via API) | ✅ Yes (via yfinance) |
| **Option Prices** | ❌ No | ✅ Yes |
| **Implied Volatility** | ❌ No | ✅ Yes (calculated from options) |
| **Authentication** | Required (API key) | Not required |
| **Rate Limits** | Free: 25/day, 5/min | Soft limits, more lenient |
| **Data Quality** | High (official API) | Good (unofficial scraping) |
| **Best For** | Historical calibration | Market calibration with options |

---

## 1. Alpha Vantage

### What You Get
- **Historical daily prices** (OHLCV + adjusted close)
- Up to 20+ years of data with `outputsize='full'`
- Clean, well-structured JSON API
- Adjusted prices for splits/dividends

### What You DON'T Get
- ❌ Option prices (calls/puts)
- ❌ Option chains
- ❌ Implied volatility surface
- ❌ Real-time market data (15-20 min delay)

### Usage
```python
from Barrier_option_inputs import EquityMarketData, HestonModelParams

# Fetch historical data
equity_data = EquityMarketData.from_alpha_vantage(
    ticker="SPY",
    api_key="YOUR_API_KEY",
    lookback_days=756,  # 3 years
    outputsize='full'
)

# Estimate Heston parameters from historical data ONLY
heston_params = HestonModelParams.estimate_from_historical(equity_data)
```

### Limitations
- **Free tier**: 25 API calls per day, 5 per minute
- No option data means you can ONLY do historical calibration
- Cannot build implied volatility surface
- Cannot calibrate Heston to market option prices

### When to Use
- ✅ You have limited daily API calls and want consistent data
- ✅ You only need historical price data
- ✅ You're doing academic/prototype work with historical calibration
- ✅ You need very clean, official data

---

## 2. Yahoo Finance (yfinance)

### What You Get
- **Historical daily prices** (OHLCV)
- **Option chains** (all strikes and expirations)
- **Bid/Ask prices** for options
- **Implied volatility** (can be calculated from option prices)
- **Volatility surface** (ITM/ATM/OTM structure)

### What You DON'T Get
- No official API (uses web scraping)
- Less reliable (can break if Yahoo changes website)
- No guaranteed uptime or SLA

### Usage

#### Method 1: Historical Data Only
```python
# Just historical prices (like Alpha Vantage)
equity_data = EquityMarketData.from_yahoo(
    ticker="SPY",
    lookback_days=756
)

heston_params = HestonModelParams.estimate_from_historical(equity_data)
```

#### Method 2: With Option Prices (RECOMMENDED for market calibration)
```python
# Historical + Option prices + Implied volatility surface
equity_data = EquityMarketData.from_yahoo_with_options(
    ticker="SPY",
    lookback_days=756,
    risk_free_rate=0.045
)

# Calibrate Heston to market option prices (BETTER than historical-only)
heston_params = HestonModelParams.calibrate_to_market(equity_data)
```

### When to Use
- ✅ You need option prices for market calibration
- ✅ You want to build a volatility surface (ITM/ATM/OTM)
- ✅ You want market-consistent Heston parameters
- ✅ You're doing production/research work that requires market data

---

## 3. Recommended Approach

### For This PFE Project

**Use both data sources strategically:**

1. **Alpha Vantage** for stable, clean historical data:
   ```python
   # Get historical prices from Alpha Vantage
   equity_data = EquityMarketData.from_alpha_vantage(
       ticker="SPY",
       api_key="A3FLPOS3EM441KDK",
       lookback_days=756
   )
   ```

2. **Yahoo Finance** for option data and market calibration:
   ```python
   # Get option prices from Yahoo
   equity_data_with_options = EquityMarketData.from_yahoo_with_options(
       ticker="SPY",
       lookback_days=756
   )

   # Calibrate to market (BETTER parameters)
   heston_params = HestonModelParams.calibrate_to_market(equity_data_with_options)
   ```

### Why This Matters for PFE

**Potential Future Exposure (PFE)** requires accurate forward-looking risk metrics.

- **Historical-only calibration** (Alpha Vantage):
  - Uses past volatility to estimate future parameters
  - May miss current market stress/fear (VIX spikes)
  - Good for: academic work, prototypes

- **Market calibration** (Yahoo with options):
  - Uses forward-looking implied volatilities
  - Captures current market expectations (risk premium, skew)
  - Better for: production risk systems, regulatory reporting
  - **Recommended for your PFE project!**

---

## 4. Code Migration Summary

### What Changed

1. **Added** `from_alpha_vantage()` method to `EquityMarketData`:
   ```python
   equity_data = EquityMarketData.from_alpha_vantage(
       ticker="SPY",
       api_key="A3FLPOS3EM441KDK",
       lookback_days=756
   )
   ```

2. **Kept** existing Yahoo Finance methods:
   - `from_yahoo()` - historical only (deprecated)
   - `from_yahoo_with_options()` - historical + options (recommended for market calibration)

3. **Created** new test script:
   - `test_alpha_vantage.py` - demonstrates Alpha Vantage usage

### Existing Test Scripts Still Work
- ✅ `test_heston_calibration.py` - uses Yahoo (historical)
- ✅ `test_vol_surface.py` - uses Yahoo (options)
- ✅ `test_market_calibration.py` - uses Yahoo (options + market calibration)

---

## 5. API Key Setup

### Alpha Vantage
1. Sign up at https://www.alphavantage.co/support/#api-key
2. Free tier: 25 requests/day, 5 requests/minute
3. Your key: `A3FLPOS3EM441KDK`

### Usage in Code
```python
# Hardcode (for testing)
equity_data = EquityMarketData.from_alpha_vantage(
    ticker="SPY",
    api_key="A3FLPOS3EM441KDK"
)

# Or use environment variable (recommended for production)
import os
equity_data = EquityMarketData.from_alpha_vantage(
    ticker="SPY",
    api_key=os.getenv("ALPHA_VANTAGE_API_KEY")
)
```

---

## 6. Ticker Mappings

| Market | Yahoo Finance | Alpha Vantage |
|--------|---------------|---------------|
| S&P 500 Index | `^GSPC` | Not available (use SPY) |
| S&P 500 ETF | `SPY` | `SPY` ✅ |
| Apple | `AAPL` | `AAPL` ✅ |
| Nasdaq 100 ETF | `QQQ` | `QQQ` ✅ |

**Note**: For S&P 500 exposure, use `SPY` (ETF) on both platforms.

---

## 7. Installation Requirements

```bash
# For Alpha Vantage
pip install requests pandas

# For Yahoo Finance
pip install yfinance

# For market calibration (implied vol calculation)
pip install scipy

# All together
pip install requests pandas yfinance scipy
```

---

## 8. Quick Reference

### Historical-Only Calibration
```python
# Option 1: Alpha Vantage (requires API key, stable)
equity_data = EquityMarketData.from_alpha_vantage("SPY", api_key="YOUR_KEY")

# Option 2: Yahoo Finance (no API key, less stable)
equity_data = EquityMarketData.from_yahoo("SPY")

# Both work with historical estimation
heston_params = HestonModelParams.estimate_from_historical(equity_data)
```

### Market Calibration (RECOMMENDED for PFE)
```python
# ONLY available via Yahoo Finance (has option data)
equity_data = EquityMarketData.from_yahoo_with_options("SPY")
heston_params = HestonModelParams.calibrate_to_market(equity_data)
```

---

## Questions?

- Alpha Vantage API docs: https://www.alphavantage.co/documentation/
- yfinance docs: https://github.com/ranaroussi/yfinance