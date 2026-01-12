# Volatility Surface Calibration Framework

Complete workflow for extracting option data and calibrating stochastic volatility models (Heston & SABR).

## Overview

This framework provides:
- **Data extraction** from Yahoo Finance Excel files
- **Model calibration** for Heston and SABR models
- **Volatility surface construction** with interpolation/extrapolation
- **QuantLib integration** for use in diffusion processes and Monte Carlo simulations

## Files

1. **yahoo_option_data_cleaner.py** - Extracts and organizes option data from Excel
2. **vol_surface_calibrator.py** - Calibrates Heston/SABR models and builds volatility surfaces
3. **example_usage.py** - Complete end-to-end example

## Quick Start

### 1. Extract Option Data

```python
from yahoo_option_data_cleaner import extract_options_data

df = extract_options_data("spx_infvol_20260109.xlsx")
# Returns DataFrame with: underlying_ticker, trading_date, maturity_date,
#                         strike, option_type, option_price, spot_price
```

### 2. Calibrate Volatility Surface

```python
from vol_surface_calibrator import calibrate_vol_surface

# Heston model
heston_result = calibrate_vol_surface(df, model='Heston')

# SABR model
sabr_result = calibrate_vol_surface(df, model='SABR')
```

### 3. Use the Surface

```python
surface = heston_result['surface']

# Get volatility for any strike/maturity
vol = surface.get_vol(strike=7000, maturity=1.5)  # 1.5 years

# Export to QuantLib
ql_surface = surface.to_quantlib()

# Visualize
surface.plot()
```

## Data Requirements

The input DataFrame needs these columns:
- `trading_date` - Date when option was traded
- `maturity_date` - Option expiration date
- `strike` - Strike price
- `option_type` - 'CALL' or 'PUT'
- `option_price` - Market price of the option
- `spot_price` - Underlying asset price

**Does the output from yahoo_option_data_cleaner.py have sufficient information?**

✓ **YES!** The cleaner provides all required fields:
- Strike prices (wide range for surface construction)
- Multiple maturities (for term structure)
- Option prices (for calibration)
- Spot prices (matched to trading dates)
- Option types (call/put for put-call parity)

The framework handles:
- Varying amounts of data (automatically interpolates/extrapolates)
- Missing data points (robust filtering)
- Different strike ranges (standardized output grid)

## Models

### Heston Model

Stochastic volatility model with mean-reverting variance:

**Parameters:**
- `v0` - Initial variance
- `kappa` - Mean reversion speed
- `theta` - Long-term variance
- `sigma_v` - Volatility of volatility
- `rho` - Correlation between asset and volatility

**Best for:** Long-dated options, consistent vol-of-vol

### SABR Model

Stochastic alpha-beta-rho model (slice-by-slice calibration):

**Parameters (per maturity):**
- `alpha` - Initial volatility level
- `beta` - CEV parameter (fixed at 0.5)
- `rho` - Correlation
- `nu` - Volatility of volatility

**Best for:** Short-dated options, capturing vol smile

## Volatility Surface Features

The `VolatilitySurface` class provides:

1. **Smooth Interpolation**
   - Uses RBF (Radial Basis Function) interpolation
   - Log-moneyness and sqrt(time) parametrization
   - Handles irregular strike/maturity grids

2. **Standardized Output**
   - Regular grid (50 strikes × N maturities)
   - Bounded volatilities (1% - 300%)
   - Works with varying input data sizes

3. **QuantLib Integration**
   - `to_quantlib()` exports `BlackVarianceSurface`
   - Compatible with Heston/SABR processes
   - Ready for Monte Carlo/PDE pricing

## Example: Using in QuantLib Diffusion

```python
import QuantLib as ql

# Calibrate surface
result = calibrate_vol_surface(df, model='Heston')
params = result['params']
spot = result['spot_price']

# Create Heston process
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
day_count = ql.Actual365Fixed()
today = ql.Date.todaysDate()

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, 0.05, day_count)
)
dividend_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, 0.0, day_count)
)

heston_process = ql.HestonProcess(
    rate_handle,
    dividend_handle,
    spot_handle,
    params['v0'],
    params['kappa'],
    params['theta'],
    params['sigma_v'],
    params['rho']
)

# Use for Monte Carlo simulation, path generation, exotic pricing, etc.
```

## Installation

Requires (install in p311-env):
```bash
pip install pandas numpy scipy QuantLib-Python matplotlib
```

## Running the Examples

```bash
# Activate your environment
source p311-env/bin/activate  # or: conda activate p311-env

# Run complete example
cd Exotic
python example_usage.py
```

This will:
1. Extract option data
2. Calibrate both Heston and SABR
3. Generate volatility surfaces
4. Export to QuantLib
5. Save results to CSV
6. Create visualization plots

## Output Files

- `spx_infvol_20260109_extracted.csv` - Cleaned option data
- `calibrated_parameters.csv` - Model parameters
- `volatility_surface_grid.csv` - Surface grid points

## Notes

- **Standardization:** The surface is always output on a regular grid, regardless of input data density
- **Extrapolation:** Uses smooth RBF extrapolation beyond observed strikes/maturities
- **Robustness:** Handles missing data, filters invalid options automatically
- **Performance:** Differential evolution for Heston (global), L-BFGS-B for SABR (local)

## Troubleshooting

**"Not enough data for calibration"**
- Need at least 20-30 option contracts
- Should have 3+ different maturities
- Filters out near-expiry options (< 7 days)

**"Calibration failed"**
- Check if option prices are reasonable
- Verify spot prices are present
- Try adjusting `min_maturity_days` parameter

**"QuantLib export failed"**
- Ensure QuantLib-Python is installed
- Check that surface has valid data (no NaNs)