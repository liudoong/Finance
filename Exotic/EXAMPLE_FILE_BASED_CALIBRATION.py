"""
Example: Using File-Based Calibration for Heston Model
========================================================

This example demonstrates how to calibrate the Heston model using local
Excel files containing option data, instead of downloading from Yahoo Finance.

This is useful when:
1. You cannot access Yahoo Finance or other APIs
2. You have your own option data files
3. You want reproducible calibration results

"""

from model_calibration import calibrate_models_from_files, RatesMarketData
from datetime import datetime

# =============================================================================
# EXAMPLE 1: Basic Usage - Calibrate from Single Excel File
# =============================================================================

print("="*80)
print("EXAMPLE 1: Basic Calibration from Excel File")
print("="*80)
print()

# Calibrate using local Excel file with option data
calibrated = calibrate_models_from_files(
    option_data_file="spx_infvol_20260109.xlsx",  # Your Excel file
    spot_price=5900.0,  # Current SPX level (provide manually)
    rates_data=None,  # Will use default USD rates
    historical_prices_file=None  # Optional: add historical prices for better calibration
)

# Extract calibrated parameters
equity_data = calibrated['equity_data']
rates_data = calibrated['rates_data']
heston_params = calibrated['heston_params']
hw_params = calibrated['hull_white_params']

print("\n" + "="*80)
print("Calibrated Heston Parameters (Ready for Use):")
print("="*80)
print(f"v0 (initial variance):    {heston_params.v0:.6f}")
print(f"theta (long-term var):    {heston_params.theta:.6f}")
print(f"kappa (mean reversion):   {heston_params.kappa:.4f}")
print(f"sigma (vol of vol):       {heston_params.sigma:.4f}")
print(f"rho (correlation):        {heston_params.rho:.4f}")
print()


# =============================================================================
# EXAMPLE 2: Advanced Usage - Custom Rate Curve
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Calibration with Custom Rate Curve")
print("="*80)
print()

# Create custom rate curve (e.g., from Bloomberg, your own data, etc.)
custom_rates = RatesMarketData(
    curve_date=datetime(2026, 1, 10),
    tenor_points=["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"],
    zero_rates=[0.0445, 0.0455, 0.0448, 0.0425, 0.0405, 0.0385, 0.0375, 0.0365],
    day_count="ACT/360",
    calendar="UnitedStates"
)

# Calibrate with custom rates
calibrated2 = calibrate_models_from_files(
    option_data_file="spx_infvol_20260109.xlsx",
    spot_price=5900.0,
    rates_data=custom_rates,  # Use custom rate curve
    historical_prices_file=None
)

print("\nCalibration with custom rates completed!")


# =============================================================================
# EXAMPLE 3: Using Calibrated Parameters in Barrier Option Inputs
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: Integrating with Barrier Option Inputs")
print("="*80)
print()

# These parameters can now be used in Barrier_option_inputs.py
# for PFE simulation and barrier option pricing

print("How to use these parameters:")
print()
print("1. In your main simulation script:")
print()
print("   from model_calibration import calibrate_models_from_files")
print("   from Barrier_option_inputs import BarrierOptionParams, SimulationParams")
print()
print("   # Load calibrated parameters")
print("   calibrated = calibrate_models_from_files(...)")
print("   heston = calibrated['heston_params']")
print("   hw = calibrated['hull_white_params']")
print()
print("2. Use for PFE simulation and barrier option pricing")
print("   - Feed into your diffusion simulation")
print("   - Calculate barrier option values")
print("   - Compute Potential Future Exposure (PFE)")
print()


# =============================================================================
# EXAMPLE 4: Different File Formats
# =============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Supported File Formats")
print("="*80)
print()

print("Your Excel file should have these columns:")
print("  - Contract Name (optional, for expiry detection)")
print("  - Strike (required)")
print("  - Implied Volatility (required, in % or decimal)")
print("  - Bid, Ask, Last Price (optional)")
print()
print("Example rows:")
print("  Contract Name       | Strike | Implied Volatility | Bid  | Ask")
print("  SPXW260112P02800000 | 2800   | 2.7031            | 0.0  | 0.05")
print("  SPXW260112P03000000 | 3000   | 2.5000            | 0.0  | 0.05")
print()
print("The code automatically:")
print("  - Converts IV from percentage to decimal if needed")
print("  - Detects expiry date from contract name")
print("  - Builds volatility surface")
print("  - Calibrates Heston parameters")
print()


# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()
print("✓ File-based calibration is now working!")
print("✓ You can use different Excel files with option data")
print("✓ The volatility surface adapts to your file format")
print("✓ Calibrated parameters are ready for PFE simulation")
print()
print("Next Steps:")
print("  1. Update your Excel files with current market data")
print("  2. Run calibration with: python model_calibration.py --files")
print("  3. Use calibrated parameters in Barrier_option_inputs.py")
print("  4. Run your PFE simulation and barrier option pricing")
print()
print("="*80)
