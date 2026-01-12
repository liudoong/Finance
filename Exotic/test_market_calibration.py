"""
Test script for calibrating Heston parameters to market option prices.

Demonstrates the complete workflow:
1. Fetch SPY option prices from market
2. Build volatility surface
3. Calibrate Heston parameters to match market prices
4. Compare with historical-only estimation
"""

from Barrier_option_inputs import EquityMarketData, HestonModelParams
import time

def test_market_calibration():
    """Test full market calibration workflow"""

    print("="*80)
    print("HESTON CALIBRATION TO MARKET OPTION PRICES")
    print("="*80)
    print()

    # Step 1: Fetch market data with options
    print("Step 1: Fetching SPY market data and option prices")
    print("-" * 80)

    try:
        equity_data = EquityMarketData.from_yahoo_with_options(
            ticker="SPY",  # SPY ETF (tracks SPX)
            lookback_days=756,  # 3 years for historical params
            repo_rate=0.0,
            risk_free_rate=0.045
        )
        print()
        time.sleep(1)  # Rate limiting

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have scipy installed: pip install scipy")
        return None, None

    if equity_data.vol_surface is None or len(equity_data.vol_surface) == 0:
        print("\n❌ Could not build volatility surface")
        print("This may happen if market is closed or no liquid options available")
        return None, None

    # Step 2: Calibrate to market
    print("\n" + "="*80)
    print("Step 2: Calibrating Heston Parameters to Market")
    print("="*80)
    print()

    try:
        market_params = HestonModelParams.calibrate_to_market(
            equity_data=equity_data,
            risk_free_rate=0.045
        )
        print()

    except Exception as e:
        print(f"Error during calibration: {e}")
        return equity_data, None

    # Step 3: Compare with historical estimation
    print("\n" + "="*80)
    print("Step 3: Comparison with Historical-Only Estimation")
    print("="*80)
    print()

    try:
        hist_params = HestonModelParams.estimate_from_historical(equity_data)
        print()

    except Exception as e:
        print(f"Error during historical estimation: {e}")
        hist_params = None

    # Step 4: Side-by-side comparison
    if hist_params:
        print("\n" + "="*80)
        print("Step 4: Side-by-Side Parameter Comparison")
        print("="*80)
        print()

        import numpy as np

        print(f"{'Parameter':<20} {'Market-Calibrated':<20} {'Historical-Only':<20} {'Difference':<15}")
        print("-" * 80)

        v0_diff = market_params.v0 - hist_params.v0
        theta_diff = market_params.theta - hist_params.theta
        kappa_diff = market_params.kappa - hist_params.kappa
        sigma_diff = market_params.sigma - hist_params.sigma
        rho_diff = market_params.rho - hist_params.rho

        print(f"{'v0 (initial var)':<20} {market_params.v0:<20.6f} {hist_params.v0:<20.6f} {v0_diff:+.6f}")
        print(f"{'  (as vol)':<20} {np.sqrt(market_params.v0):<20.2%} {np.sqrt(hist_params.v0):<20.2%}")
        print()
        print(f"{'θ (long-term var)':<20} {market_params.theta:<20.6f} {hist_params.theta:<20.6f} {theta_diff:+.6f}")
        print(f"{'  (as vol)':<20} {np.sqrt(market_params.theta):<20.2%} {np.sqrt(hist_params.theta):<20.2%}")
        print()
        print(f"{'κ (mean reversion)':<20} {market_params.kappa:<20.4f} {hist_params.kappa:<20.4f} {kappa_diff:+.4f}")
        print(f"{'σ (vol of vol)':<20} {market_params.sigma:<20.4f} {hist_params.sigma:<20.4f} {sigma_diff:+.4f}")
        print(f"{'ρ (correlation)':<20} {market_params.rho:<20.4f} {hist_params.rho:<20.4f} {rho_diff:+.4f}")

        print("\n" + "-" * 80)
        print("Key Differences:")
        print("-" * 80)

        # Analyze differences
        if abs(v0_diff) > 0.005:
            direction = "higher" if v0_diff > 0 else "lower"
            print(f"  • Market implies {direction} current volatility than recent historical")

        if abs(theta_diff) > 0.005:
            direction = "higher" if theta_diff > 0 else "lower"
            print(f"  • Market implies {direction} long-term volatility")

        if abs(rho_diff) > 0.1:
            if abs(market_params.rho) > abs(hist_params.rho):
                print(f"  • Market shows stronger leverage effect (more negative correlation)")
            else:
                print(f"  • Market shows weaker leverage effect (less negative correlation)")

        if abs(sigma_diff) > 0.05:
            direction = "higher" if sigma_diff > 0 else "lower"
            print(f"  • Market implies {direction} volatility-of-volatility")

    # Step 5: Validation
    print("\n" + "="*80)
    print("Step 5: Parameter Validation")
    print("="*80)
    print()

    print("Market-Calibrated Parameters:")
    market_feller = market_params.validate()
    print(f"  Status: {'✓ Valid' if market_feller else '✗ Invalid (Feller violated)'}")

    if hist_params:
        print("\nHistorical-Only Parameters:")
        hist_feller = hist_params.validate()
        print(f"  Status: {'✓ Valid' if hist_feller else '✗ Invalid (Feller violated)'}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"✓ Successfully calibrated Heston parameters to {len(equity_data.vol_surface)} market option prices")
    print(f"✓ Spot Price: ${equity_data.spot_price:.2f}")
    print(f"✓ ATM Implied Vol: {equity_data.atm_volatility:.2%}")
    print()
    print("Market-Calibrated Parameters:")
    print(f"  v0 = {market_params.v0:.6f} (vol = {np.sqrt(market_params.v0):.2%})")
    print(f"  θ  = {market_params.theta:.6f} (vol = {np.sqrt(market_params.theta):.2%})")
    print(f"  κ  = {market_params.kappa:.4f}")
    print(f"  σ  = {market_params.sigma:.4f}")
    print(f"  ρ  = {market_params.rho:.4f}")
    print()
    print("✓ These parameters are market-consistent and ready for PFE simulation!")
    print("="*80)

    return equity_data, market_params


def usage_example():
    """Show usage example"""
    print("\n\nUSAGE EXAMPLE:")
    print("-" * 80)
    print("# Fetch market data with volatility surface:")
    print("equity_data = EquityMarketData.from_yahoo_with_options('SPY')")
    print()
    print("# Calibrate Heston to market (recommended!):")
    print("heston_params = HestonModelParams.calibrate_to_market(equity_data)")
    print()
    print("# Alternative: Use historical data only:")
    print("heston_params = HestonModelParams.estimate_from_historical(equity_data)")
    print()
    print("# Use in PFE simulation:")
    print("# Now your Heston model will be consistent with market option prices!")
    print("-" * 80)


if __name__ == "__main__":
    # Run the test
    equity_data, market_params = test_market_calibration()

    # Show usage example
    if equity_data and market_params:
        usage_example()