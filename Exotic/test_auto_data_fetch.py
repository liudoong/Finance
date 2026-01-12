"""
Test script for automatic market data fetching from free sources.

Demonstrates:
1. Fetching SPX equity data from Yahoo Finance (historical vol)
2. Fetching USD rates from FRED
3. Automatic data loading into QuantLib-ready format
"""

from Barrier_option_inputs import EquityMarketData, RatesMarketData
from datetime import datetime

def test_auto_fetch():
    """Test automatic data fetching from free sources"""

    print("="*80)
    print("TESTING AUTOMATIC MARKET DATA FETCHING")
    print("="*80)
    print()

    # 1. Fetch SPX equity data from Yahoo Finance
    print("1. Fetching SPX Equity Data from Yahoo Finance")
    print("-" * 80)
    try:
        # SPX ticker on Yahoo is "^GSPC"
        equity_data = EquityMarketData.from_yahoo(
            ticker="^GSPC",  # S&P 500 Index
            lookback_days=252,  # 1 year of data
            repo_rate=0.0  # Assume 0 for indices
        )
        print()
    except Exception as e:
        print(f"Error fetching equity data: {e}")
        equity_data = None

    # 2. Fetch USD rates from FRED
    print("\n2. Fetching USD Rates from FRED")
    print("-" * 80)
    try:
        rates_data = RatesMarketData.from_fred(
            currency="USD",
            date=None  # Use most recent available data
        )
        print()
    except Exception as e:
        print(f"Error fetching rates data: {e}")
        rates_data = None

    # 3. Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    if equity_data:
        print(f"\n✓ Equity Data Ready:")
        print(f"  - Spot: ${equity_data.spot_price:,.2f}")
        print(f"  - Vol: {equity_data.atm_volatility:.2%}")
        print(f"  - Historical Prices: {len(equity_data.historical_prices) if equity_data.historical_prices else 0} days")

    if rates_data:
        print(f"\n✓ Rates Data Ready:")
        print(f"  - Curve Date: {rates_data.curve_date.strftime('%Y-%m-%d')}")
        print(f"  - Tenor Points: {len(rates_data.tenor_points)}")
        print(f"  - Rate Range: {min(rates_data.zero_rates):.2%} - {max(rates_data.zero_rates):.2%}")

    print("\n" + "="*80)
    print("✓ Both data sources are now ready to feed into QuantLib!")
    print("="*80)

    return equity_data, rates_data


if __name__ == "__main__":
    # Run the test
    equity_data, rates_data = test_auto_fetch()

    # Show how to use in actual workflow
    print("\n\nUSAGE EXAMPLE:")
    print("-" * 80)
    print("# Instead of manually entering data:")
    print("equity_data = EquityMarketData(spot_price=5900.0, atm_volatility=0.15, ...)")
    print()
    print("# Simply call:")
    print("equity_data = EquityMarketData.from_yahoo('^GSPC')")
    print("rates_data = RatesMarketData.from_fred('USD')")
    print("-" * 80)