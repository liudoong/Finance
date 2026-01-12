"""
Test script for fetching volatility surface from option prices.

Demonstrates:
1. Fetching SPY option prices from Yahoo Finance (free)
2. Calculating implied volatilities
3. Building volatility surface with ITM/ATM/OTM structure
4. Comparing historical vol vs implied vol
"""

from Barrier_option_inputs import EquityMarketData
import time

def test_vol_surface():
    """Test volatility surface construction from market option prices"""

    print("="*80)
    print("VOLATILITY SURFACE FROM OPTION PRICES")
    print("="*80)
    print()

    print("Note: Using SPY (S&P 500 ETF) because ^GSPC doesn't have listed options")
    print("      SPY tracks SPX very closely and has liquid options")
    print()

    # Step 1: Fetch market data WITH volatility surface
    print("Step 1: Fetching SPY market data and option prices")
    print("-" * 80)

    try:
        equity_data = EquityMarketData.from_yahoo_with_options(
            ticker="SPY",  # SPY ETF (tracks SPX)
            lookback_days=756,  # 3 years for historical params
            repo_rate=0.0,
            risk_free_rate=0.045  # 4.5% for IV calculation
        )
        print()
        time.sleep(1)  # Rate limiting

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have scipy installed: pip install scipy")
        return None

    # Step 2: Analyze the volatility surface
    print("\nStep 2: Analyzing Volatility Surface")
    print("-" * 80)

    if equity_data.vol_surface is None:
        print("Warning: No volatility surface was built")
        print("This may happen if:")
        print("  - Market is closed")
        print("  - No liquid options available")
        print("  - Data fetch failed")
        return equity_data

    spot = equity_data.spot_price
    atm_vol = equity_data.atm_volatility

    print(f"\nSpot Price: ${spot:.2f}")
    print(f"ATM Implied Vol: {atm_vol:.2%}")
    print(f"\nVolatility Surface Points: {len(equity_data.vol_surface)}")

    # Categorize options by moneyness
    itm_puts = []  # Strike > Spot (in the money for puts)
    atm_options = []  # Strike ≈ Spot
    otm_puts = []  # Strike < Spot (out of money for puts)

    for (strike, tenor), iv in equity_data.vol_surface.items():
        moneyness = strike / spot

        if moneyness > 1.05:  # More than 5% ITM
            itm_puts.append((strike, tenor, iv, moneyness))
        elif 0.95 <= moneyness <= 1.05:  # Within 5% of ATM
            atm_options.append((strike, tenor, iv, moneyness))
        else:  # moneyness < 0.95, OTM
            otm_puts.append((strike, tenor, iv, moneyness))

    print(f"\nBreakdown by Moneyness (for PUT options):")
    print(f"  ITM Puts (Strike > Spot):  {len(itm_puts)} points")
    print(f"  ATM (Strike ≈ Spot):       {len(atm_options)} points")
    print(f"  OTM Puts (Strike < Spot):  {len(otm_puts)} points")

    # Show volatility smile/skew
    print(f"\nVolatility Smile/Skew Analysis:")
    print(f"{'Moneyness':<15} {'Strike':<10} {'Tenor':<10} {'Impl Vol':<10}")
    print("-" * 80)

    # Sample from each category
    if itm_puts:
        avg_itm_vol = sum(iv for _, _, iv, _ in itm_puts) / len(itm_puts)
        strike, tenor, iv, m = itm_puts[0]
        print(f"{'ITM (>105%)':<15} {strike:<10.0f} {tenor:<10} {iv:<10.2%}  (avg: {avg_itm_vol:.2%})")

    if atm_options:
        avg_atm_vol = sum(iv for _, _, iv, _ in atm_options) / len(atm_options)
        strike, tenor, iv, m = atm_options[0]
        print(f"{'ATM (95-105%)':<15} {strike:<10.0f} {tenor:<10} {iv:<10.2%}  (avg: {avg_atm_vol:.2%})")

    if otm_puts:
        avg_otm_vol = sum(iv for _, _, iv, _ in otm_puts) / len(otm_puts)
        strike, tenor, iv, m = otm_puts[0]
        print(f"{'OTM (<95%)':<15} {strike:<10.0f} {tenor:<10} {iv:<10.2%}  (avg: {avg_otm_vol:.2%})")

    # Volatility skew observation
    if itm_puts and otm_puts:
        avg_itm_vol = sum(iv for _, _, iv, _ in itm_puts) / len(itm_puts)
        avg_otm_vol = sum(iv for _, _, iv, _ in otm_puts) / len(otm_puts)
        skew = avg_itm_vol - avg_otm_vol

        print(f"\n✓ Volatility Skew Detected:")
        print(f"  ITM puts have {skew:+.2%} higher vol than OTM puts")
        if skew > 0:
            print(f"  This is typical 'volatility smile/skew' for equity index options")

    # Step 3: Compare with historical volatility
    print(f"\n\nStep 3: Historical vs Implied Volatility")
    print("-" * 80)

    if equity_data.historical_prices:
        import numpy as np
        prices = np.array(equity_data.historical_prices)
        returns = np.diff(np.log(prices))

        # Calculate historical vol over different periods
        hist_1m = np.std(returns[-21:]) * np.sqrt(252)  # Last month
        hist_3m = np.std(returns[-63:]) * np.sqrt(252)  # Last 3 months
        hist_1y = np.std(returns[-252:]) * np.sqrt(252)  # Last year

        print(f"Historical Volatility:")
        print(f"  1-Month:  {hist_1m:.2%}")
        print(f"  3-Month:  {hist_3m:.2%}")
        print(f"  1-Year:   {hist_1y:.2%}")
        print(f"\nImplied ATM Vol: {atm_vol:.2%}")
        print(f"\nDifference (Implied - Historical 3M): {atm_vol - hist_3m:+.2%}")

        if atm_vol > hist_3m + 0.02:
            print("  → Market is pricing in HIGHER future volatility (elevated fear/uncertainty)")
        elif atm_vol < hist_3m - 0.02:
            print("  → Market is pricing in LOWER future volatility (complacency)")
        else:
            print("  → Implied vol roughly matches recent historical vol")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Successfully built volatility surface from market option prices")
    print(f"✓ Surface contains {len(equity_data.vol_surface)} (strike, tenor) points")
    print(f"✓ Captured volatility smile/skew structure")
    print(f"✓ ATM Implied Vol: {atm_vol:.2%}")
    print()
    print("This volatility surface can now be used for:")
    print("  1. Heston parameter calibration (better than historical-only)")
    print("  2. Validating model prices against market")
    print("  3. Risk analysis with market-consistent volatility")
    print("="*80)

    return equity_data


def usage_example():
    """Show usage example"""
    print("\n\nUSAGE EXAMPLE:")
    print("-" * 80)
    print("# Fetch with volatility surface (recommended for calibration):")
    print("equity_data = EquityMarketData.from_yahoo_with_options('SPY')")
    print()
    print("# Access the surface:")
    print("for (strike, tenor), iv in equity_data.vol_surface.items():")
    print("    print(f'Strike {strike}, Tenor {tenor}: {iv:.2%}')")
    print()
    print("# Use for Heston calibration:")
    print("heston_params = HestonModelParams.calibrate_to_market(")
    print("    equity_data=equity_data,")
    print("    vol_surface=equity_data.vol_surface")
    print(")")
    print("-" * 80)


if __name__ == "__main__":
    # Run the test
    equity_data = test_vol_surface()

    # Show usage example
    if equity_data:
        usage_example()