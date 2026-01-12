"""
Test script for fetching data from Alpha Vantage API.

Demonstrates:
1. Fetching historical data from Alpha Vantage
2. Estimating Heston parameters from historical data
3. Comparing with manual parameters
"""

from Barrier_option_inputs import EquityMarketData, HestonModelParams
import time

# Your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "A3FLPOS3EM441KDK"

def test_alpha_vantage():
    """Test Alpha Vantage data fetching and Heston estimation"""

    print("="*80)
    print("ALPHA VANTAGE DATA FETCHING AND HESTON ESTIMATION")
    print("="*80)
    print()

    # Step 1: Fetch historical data from Alpha Vantage
    print("Step 1: Fetching historical data from Alpha Vantage")
    print("-" * 80)

    try:
        equity_data = EquityMarketData.from_alpha_vantage(
            ticker="SPY",  # SPY ETF (use instead of ^GSPC which doesn't work on Alpha Vantage)
            api_key=ALPHA_VANTAGE_API_KEY,
            lookback_days=756,  # 3 years of data for better parameter estimation
            repo_rate=0.0,
            outputsize='full'  # Get full historical data (not just 100 days)
        )
        print()
        time.sleep(1)  # Rate limiting

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Check your API key and rate limits")
        print("Alpha Vantage free tier limits:")
        print("  - 25 requests per day")
        print("  - 5 requests per minute")
        return None, None

    # Step 2: Estimate Heston parameters
    print("\nStep 2: Estimating Heston Parameters")
    print("-" * 80)

    try:
        heston_params = HestonModelParams.estimate_from_historical(equity_data)
        print()
    except Exception as e:
        print(f"Error: {e}")
        return equity_data, None

    # Step 3: Validate Feller condition
    print("\nStep 3: Validating Feller Condition")
    print("-" * 80)
    is_valid = heston_params.validate()
    print(f"Feller condition: {'✓ SATISFIED' if is_valid else '✗ VIOLATED'}")
    print()

    # Step 4: Compare with manual parameters
    print("\nStep 4: Comparison with Manual (Textbook) Parameters")
    print("-" * 80)
    manual_params = HestonModelParams(
        v0=0.0225,
        theta=0.0225,
        kappa=2.0,
        sigma=0.3,
        rho=-0.7,
        calibrated=False
    )

    import numpy as np

    print(f"{'Parameter':<20} {'Estimated':<15} {'Manual':<15} {'Difference':<15}")
    print("-" * 80)
    print(f"{'v0 (initial var)':<20} {heston_params.v0:<15.6f} {manual_params.v0:<15.6f} {heston_params.v0 - manual_params.v0:+.6f}")
    print(f"{'  (as vol)':<20} {np.sqrt(heston_params.v0):<15.2%} {np.sqrt(manual_params.v0):<15.2%}")
    print()
    print(f"{'θ (long-term var)':<20} {heston_params.theta:<15.6f} {manual_params.theta:<15.6f} {heston_params.theta - manual_params.theta:+.6f}")
    print(f"{'  (as vol)':<20} {np.sqrt(heston_params.theta):<15.2%} {np.sqrt(manual_params.theta):<15.2%}")
    print()
    print(f"{'κ (mean reversion)':<20} {heston_params.kappa:<15.4f} {manual_params.kappa:<15.4f} {heston_params.kappa - manual_params.kappa:+.4f}")
    print(f"{'σ (vol of vol)':<20} {heston_params.sigma:<15.4f} {manual_params.sigma:<15.4f} {heston_params.sigma - manual_params.sigma:+.4f}")
    print(f"{'ρ (correlation)':<20} {heston_params.rho:<15.4f} {manual_params.rho:<15.4f} {heston_params.rho - manual_params.rho:+.4f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Heston parameters estimated from {len(equity_data.historical_prices)} days of data")
    print(f"✓ Data source: Alpha Vantage API")
    print(f"✓ Ticker: SPY")
    print(f"✓ Calibration status: {heston_params.calibrated}")
    print(f"✓ Feller condition: {'Satisfied' if is_valid else 'Violated (adjust parameters)'}")
    print()
    print("These parameters are now ready for use in PFE simulation!")
    print("="*80)

    return equity_data, heston_params


def usage_example():
    """Show usage example"""
    print("\n\nUSAGE EXAMPLE:")
    print("-" * 80)
    print("# Alpha Vantage data fetching (recommended with your API key):")
    print("equity_data = EquityMarketData.from_alpha_vantage(")
    print("    ticker='SPY',")
    print("    api_key='YOUR_API_KEY',")
    print("    lookback_days=756")
    print(")")
    print()
    print("# Estimate Heston parameters:")
    print("heston_params = HestonModelParams.estimate_from_historical(equity_data)")
    print()
    print("# Alternative: Manual specification:")
    print("heston_params = HestonModelParams(")
    print("    v0=0.0225, theta=0.0225, kappa=2.0,")
    print("    sigma=0.3, rho=-0.7, calibrated=False")
    print(")")
    print()
    print("# Note: Alpha Vantage does NOT provide option prices")
    print("# For option data and market calibration, use Yahoo Finance:")
    print("equity_data = EquityMarketData.from_yahoo_with_options('SPY')")
    print("heston_params = HestonModelParams.calibrate_to_market(equity_data)")
    print("-" * 80)


if __name__ == "__main__":
    # Run the test
    equity_data, heston_params = test_alpha_vantage()

    # Show usage example
    if equity_data and heston_params:
        usage_example()