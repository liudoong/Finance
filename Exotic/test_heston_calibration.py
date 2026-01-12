"""
Test script for Heston parameter estimation from historical data.

Demonstrates:
1. Fetching SPX historical data from Yahoo Finance
2. Automatically estimating Heston parameters
3. Comparing with manual parameters
"""

from Barrier_option_inputs import EquityMarketData, HestonModelParams
import time

def test_heston_estimation():
    """Test automatic Heston parameter estimation"""

    print("="*80)
    print("HESTON PARAMETER ESTIMATION FROM HISTORICAL DATA")
    print("="*80)
    print()

    # Step 1: Fetch historical data from Yahoo Finance
    print("Step 1: Fetching historical data from Yahoo Finance")
    print("-" * 80)
    try:
        equity_data = EquityMarketData.from_yahoo(
            ticker="^GSPC",  # S&P 500
            lookback_days=756,  # 3 years of data (252 * 3) for better parameter estimation
            repo_rate=0.0
        )
        print()
        time.sleep(1)  # Rate limiting
    except Exception as e:
        print(f"Error: {e}")
        print("Note: If rate limited, try again in a few moments")
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

    print(f"{'Parameter':<20} {'Estimated':<15} {'Manual':<15} {'Difference':<15}")
    print("-" * 80)
    print(f"{'v0 (initial var)':<20} {heston_params.v0:<15.6f} {manual_params.v0:<15.6f} {heston_params.v0 - manual_params.v0:+.6f}")
    print(f"{'θ (long-term var)':<20} {heston_params.theta:<15.6f} {manual_params.theta:<15.6f} {heston_params.theta - manual_params.theta:+.6f}")
    print(f"{'κ (mean reversion)':<20} {heston_params.kappa:<15.4f} {manual_params.kappa:<15.4f} {heston_params.kappa - manual_params.kappa:+.4f}")
    print(f"{'σ (vol of vol)':<20} {heston_params.sigma:<15.4f} {manual_params.sigma:<15.4f} {heston_params.sigma - manual_params.sigma:+.4f}")
    print(f"{'ρ (correlation)':<20} {heston_params.rho:<15.4f} {manual_params.rho:<15.4f} {heston_params.rho - manual_params.rho:+.4f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Heston parameters estimated from {len(equity_data.historical_prices)} days of data")
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
    print("# Automatic estimation (recommended):")
    print("equity_data = EquityMarketData.from_yahoo('^GSPC')")
    print("heston_params = HestonModelParams.estimate_from_historical(equity_data)")
    print()
    print("# Manual specification (if you know the parameters):")
    print("heston_params = HestonModelParams(")
    print("    v0=0.0225, theta=0.0225, kappa=2.0,")
    print("    sigma=0.3, rho=-0.7, calibrated=False")
    print(")")
    print("-" * 80)


if __name__ == "__main__":
    # Run the test
    equity_data, heston_params = test_heston_estimation()

    # Show usage example
    if equity_data and heston_params:
        usage_example()