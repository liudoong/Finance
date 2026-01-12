"""
Download SPX Option Data for Heston Model Calibration

This script downloads SPX option price data from Yahoo Finance and prepares it
for Heston model calibration. The output file contains option chains with
strikes, prices, implied volatilities, and other market data.

Usage:
    python download_spx_options.py

Output:
    - CSV file with SPX option data
    - Includes both calls and puts
    - Ready for calibration in model_calibration.py
"""

from yahoo_option_data import download_option_data, get_available_expirations, save_option_data
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np


def download_spx_for_calibration(
    ticker: str = "SPY",
    num_expirations: int = 3,
    output_filename: str = None
):
    """
    Download SPX option data formatted for Heston calibration.

    Parameters:
    -----------
    ticker : str
        SPX ticker symbol. Options:
        - "^SPX" : S&P 500 Index (may have limited data on Yahoo)
        - "SPY" : S&P 500 ETF (recommended - more liquid, better data)
    num_expirations : int
        Number of nearest expiration dates to download
    output_filename : str, optional
        Custom output filename

    Returns:
    --------
    pd.DataFrame
        Option data with calibration-ready format
    """
    print("="*80)
    print("SPX OPTION DATA DOWNLOAD FOR HESTON CALIBRATION")
    print("="*80)
    print(f"Ticker: {ticker}")
    print(f"Number of expirations: {num_expirations}")
    print()

    # Get available expiration dates
    print("Step 1: Getting available expiration dates...")
    try:
        expirations = get_available_expirations(ticker)
        print(f"  Found {len(expirations)} available expiration dates")
        print(f"  Nearest expirations: {expirations[:num_expirations]}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        print()
        print("  Note: ^SPX may have limited data on Yahoo Finance.")
        print("  Recommendation: Use 'SPY' instead for better data quality.")
        return None

    print()

    # Get current spot price
    print("Step 2: Getting current spot price...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if not hist.empty:
        spot_price = hist['Close'].iloc[-1]
        print(f"  Current spot price: ${spot_price:.2f}")
    else:
        print("  Warning: Could not fetch spot price")
        spot_price = None

    print()

    # Download option data for multiple expirations
    print(f"Step 3: Downloading option chains for {num_expirations} expirations...")
    all_options = []

    for i, exp_date in enumerate(expirations[:num_expirations], 1):
        print(f"  [{i}/{num_expirations}] Downloading {exp_date}...")
        try:
            data = download_option_data(
                ticker=ticker,
                option_type='both',  # Both calls and puts for calibration
                expiration_date=exp_date
            )
            all_options.append(data)
            print(f"      Retrieved {len(data)} contracts")
        except Exception as e:
            print(f"      Error: {str(e)}")

    if not all_options:
        print("\n  Error: No option data downloaded")
        return None

    # Combine all data
    print()
    print("Step 4: Combining and processing data...")
    combined_data = pd.concat(all_options, ignore_index=True)
    print(f"  Total contracts: {len(combined_data)}")

    # Add derived fields useful for calibration
    if spot_price:
        combined_data['moneyness'] = combined_data['strike'] / spot_price
        combined_data['spotPrice'] = spot_price

        # Filter to reasonable moneyness range (0.8 to 1.2 = 80% to 120%)
        # This focuses on ATM and near-ATM options which are most important for calibration
        mask = (combined_data['moneyness'] >= 0.8) & (combined_data['moneyness'] <= 1.2)
        filtered_data = combined_data[mask].copy()
        print(f"  Filtered to moneyness 0.8-1.2: {len(filtered_data)} contracts")
    else:
        filtered_data = combined_data

    # Calculate time to expiration in years
    filtered_data['daysToExpiration'] = pd.to_datetime(filtered_data['expirationDate']) - datetime.now()
    filtered_data['daysToExpiration'] = filtered_data['daysToExpiration'].dt.days
    filtered_data['timeToExpiration'] = filtered_data['daysToExpiration'] / 365.25

    # Select important columns for calibration
    calibration_columns = [
        'ticker',
        'expirationDate',
        'daysToExpiration',
        'timeToExpiration',
        'optionType',
        'strike',
        'lastPrice',
        'bid',
        'ask',
        'volume',
        'openInterest',
        'impliedVolatility',
        'moneyness',
        'spotPrice',
        'downloadTime'
    ]

    # Keep only available columns
    available_cols = [col for col in calibration_columns if col in filtered_data.columns]
    final_data = filtered_data[available_cols]

    # Remove contracts with zero or missing implied volatility
    if 'impliedVolatility' in final_data.columns:
        initial_count = len(final_data)
        final_data = final_data[final_data['impliedVolatility'] > 0].copy()
        removed = initial_count - len(final_data)
        if removed > 0:
            print(f"  Removed {removed} contracts with zero/missing IV")

    print()
    print("Step 5: Data Summary")
    print("-" * 80)
    print(f"  Total contracts: {len(final_data)}")
    if 'optionType' in final_data.columns:
        print(f"  Calls: {len(final_data[final_data['optionType'] == 'call'])}")
        print(f"  Puts: {len(final_data[final_data['optionType'] == 'put'])}")
    if 'timeToExpiration' in final_data.columns:
        print(f"  Time to expiration range: {final_data['timeToExpiration'].min():.4f} to {final_data['timeToExpiration'].max():.4f} years")
    if 'impliedVolatility' in final_data.columns:
        print(f"  IV range: {final_data['impliedVolatility'].min():.2%} to {final_data['impliedVolatility'].max():.2%}")
        print(f"  ATM IV: ~{final_data['impliedVolatility'].median():.2%}")

    print()

    # Save to file
    print("Step 6: Saving to file...")
    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ticker_clean = ticker.replace('^', '')
        output_filename = f"{ticker_clean}_options_calibration_{timestamp}.csv"

    final_data.to_csv(output_filename, index=False)
    print(f"  Saved to: {output_filename}")

    print()
    print("="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print()
    print("Sample data (first 10 rows):")
    print(final_data.head(10).to_string())
    print()
    print(f"Next step: Use '{output_filename}' for Heston model calibration")
    print()

    return final_data


if __name__ == "__main__":
    # Download SPX option data
    # Note: ^SPX may have limited data on Yahoo Finance
    # SPY (S&P 500 ETF) is recommended as an alternative

    print("Attempting to download ^SPX data...")
    print("If this fails, we'll try SPY as an alternative.")
    print()

    data = download_spx_for_calibration(
        ticker="^SPX",
        num_expirations=3
    )

    # If ^SPX fails, try SPY
    if data is None or len(data) == 0:
        print()
        print("="*80)
        print("^SPX data unavailable. Trying SPY (S&P 500 ETF) instead...")
        print("="*80)
        print()

        data = download_spx_for_calibration(
            ticker="SPY",
            num_expirations=3
        )

    if data is not None:
        print("\n✓ SUCCESS: Option data ready for calibration!")
    else:
        print("\n✗ FAILED: Could not download option data")