"""
Example: Complete workflow from data extraction to calibrated volatility surface

This demonstrates:
1. Extract options data from Excel
2. Calibrate Heston or SABR model
3. Generate standardized volatility surface
4. Export to QuantLib for use in diffusion processes
"""

from yahoo_option_data_cleaner import extract_options_data
from vol_surface_calibrator import calibrate_vol_surface
import pandas as pd


def main():
    # Step 1: Extract and clean option data
    print("=" * 70)
    print("STEP 1: Extract Option Data")
    print("=" * 70)

    input_file = "spx_infvol_20260109.xlsx"
    df = extract_options_data(input_file)

    print(f"\nExtracted {len(df)} option contracts")
    print(f"\nSample data:")
    print(df.head())

    # Step 2: Calibrate Heston model
    print("\n\n" + "=" * 70)
    print("STEP 2: Calibrate Heston Model")
    print("=" * 70)

    heston_result = calibrate_vol_surface(
        df,
        model='Heston',
        risk_free_rate=0.05,
        min_maturity_days=7
    )

    heston_surface = heston_result['surface']
    heston_params = heston_result['params']

    # Step 3: Calibrate SABR model
    print("\n\n" + "=" * 70)
    print("STEP 3: Calibrate SABR Model")
    print("=" * 70)

    sabr_result = calibrate_vol_surface(
        df,
        model='SABR',
        risk_free_rate=0.05,
        min_maturity_days=7
    )

    sabr_surface = sabr_result['surface']
    sabr_params = sabr_result['params']

    # Step 4: Use the calibrated surfaces
    print("\n\n" + "=" * 70)
    print("STEP 4: Using Calibrated Surfaces")
    print("=" * 70)

    # Example: Get volatility for specific strikes/maturities
    spot = heston_result['spot_price']

    test_cases = [
        (spot * 0.9, 0.5, "6-month 90% OTM Put"),
        (spot, 1.0, "1-year ATM"),
        (spot * 1.1, 2.0, "2-year 110% OTM Call"),
    ]

    print("\nHeston Model Volatilities:")
    for strike, maturity, description in test_cases:
        vol = heston_surface.get_vol(strike, maturity)
        print(f"  {description:30s}: {vol:.2%}")

    print("\nSABR Model Volatilities:")
    for strike, maturity, description in test_cases:
        vol = sabr_surface.get_vol(strike, maturity)
        print(f"  {description:30s}: {vol:.2%}")

    # Step 5: Export to QuantLib (if available)
    print("\n\n" + "=" * 70)
    print("STEP 5: Export to QuantLib")
    print("=" * 70)

    try:
        import QuantLib as ql

        # Export Heston surface
        heston_ql_surface = heston_surface.to_quantlib()
        print("✓ Heston surface exported to QuantLib successfully")

        # You can now use this surface in QuantLib diffusion processes
        # Example: Create Heston process for Monte Carlo
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

        # Create Heston process using calibrated parameters
        heston_process = ql.HestonProcess(
            rate_handle,
            dividend_handle,
            spot_handle,
            heston_params['v0'],
            heston_params['kappa'],
            heston_params['theta'],
            heston_params['sigma_v'],
            heston_params['rho']
        )

        print("✓ Heston process created and ready for diffusion/Monte Carlo")

        # You can now use heston_process for:
        # - Monte Carlo simulation
        # - Path generation
        # - Exotic option pricing
        # - Risk analysis

    except ImportError:
        print("✗ QuantLib not available")
        print("  Install with: pip install QuantLib-Python")

    # Step 6: Visualize (if matplotlib available)
    print("\n\n" + "=" * 70)
    print("STEP 6: Visualize Surfaces")
    print("=" * 70)

    try:
        print("\nGenerating Heston surface plot...")
        heston_surface.plot()

        print("\nGenerating SABR surface plot...")
        sabr_surface.plot()
    except ImportError:
        print("✗ Matplotlib not available for plotting")

    # Step 7: Save results
    print("\n\n" + "=" * 70)
    print("STEP 7: Save Results")
    print("=" * 70)

    # Save calibration parameters
    params_df = pd.DataFrame([
        {'Model': 'Heston', **heston_params},
        {'Model': 'SABR', 'params': str(sabr_params)}
    ])
    params_df.to_csv('calibrated_parameters.csv', index=False)
    print("✓ Parameters saved to calibrated_parameters.csv")

    # Save surface grid
    strikes = heston_surface.strikes
    maturities = heston_surface.maturities

    surface_grid = []
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            surface_grid.append({
                'Strike': K,
                'Maturity': T,
                'Heston_Vol': heston_surface.implied_vols[i, j],
                'SABR_Vol': sabr_surface.implied_vols[i, j]
            })

    grid_df = pd.DataFrame(surface_grid)
    grid_df.to_csv('volatility_surface_grid.csv', index=False)
    print("✓ Surface grid saved to volatility_surface_grid.csv")

    print("\n\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nYou now have:")
    print("  1. Calibrated Heston and SABR models")
    print("  2. Smooth volatility surfaces with interpolation")
    print("  3. QuantLib-compatible objects for diffusion processes")
    print("  4. Saved parameters and surface grids")


if __name__ == "__main__":
    main()