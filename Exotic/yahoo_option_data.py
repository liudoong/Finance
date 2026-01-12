"""
Yahoo Finance Option Data Downloader

A simple, independent module for downloading equity option price data from Yahoo Finance.
Can be run standalone or imported into other scripts.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional, Literal


def download_option_data(
    ticker: str,
    option_type: Literal['call', 'put', 'both'] = 'both',
    expiration_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Download option data for a given ticker from Yahoo Finance.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    option_type : str, default 'both'
        Type of options to download: 'call', 'put', or 'both'
    expiration_date : str, optional
        Specific expiration date in 'YYYY-MM-DD' format.
        If None, downloads the nearest expiration date.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing option data with columns like strike, bid, ask,
        last price, volume, open interest, and implied volatility.

    Example:
    --------
    >>> data = download_option_data('AAPL', option_type='call')
    >>> print(data.head())
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)

        # Get available expiration dates
        expirations = stock.options

        if len(expirations) == 0:
            raise ValueError(f"No options available for {ticker}")

        # Use specified expiration or default to first available
        if expiration_date:
            if expiration_date not in expirations:
                raise ValueError(f"Expiration date {expiration_date} not available. "
                               f"Available dates: {expirations}")
            exp_date = expiration_date
        else:
            exp_date = expirations[0]

        # Get option chain
        opt_chain = stock.option_chain(exp_date)

        # Extract requested option type(s)
        if option_type == 'call':
            data = opt_chain.calls.copy()
            data['optionType'] = 'call'
        elif option_type == 'put':
            data = opt_chain.puts.copy()
            data['optionType'] = 'put'
        else:  # both
            calls = opt_chain.calls.copy()
            calls['optionType'] = 'call'
            puts = opt_chain.puts.copy()
            puts['optionType'] = 'put'
            data = pd.concat([calls, puts], ignore_index=True)

        # Add metadata
        data['ticker'] = ticker
        data['expirationDate'] = exp_date
        data['downloadTime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return data

    except Exception as e:
        raise Exception(f"Error downloading option data for {ticker}: {str(e)}")


def get_available_expirations(ticker: str) -> list:
    """
    Get all available expiration dates for a ticker.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol

    Returns:
    --------
    list
        List of expiration dates as strings
    """
    try:
        stock = yf.Ticker(ticker)
        return list(stock.options)
    except Exception as e:
        raise Exception(f"Error getting expirations for {ticker}: {str(e)}")


def save_option_data(
    data: pd.DataFrame,
    filename: Optional[str] = None,
    format: Literal['csv', 'excel'] = 'csv'
) -> str:
    """
    Save option data to file.

    Parameters:
    -----------
    data : pd.DataFrame
        Option data DataFrame
    filename : str, optional
        Output filename. If None, auto-generates based on ticker and timestamp
    format : str, default 'csv'
        Output format: 'csv' or 'excel'

    Returns:
    --------
    str
        Path to saved file
    """
    if filename is None:
        ticker = data['ticker'].iloc[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_options_{timestamp}.{format}"

    if format == 'csv':
        data.to_csv(filename, index=False)
    elif format == 'excel':
        data.to_excel(filename, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return filename


# Main execution for standalone use
if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"

    print(f"Downloading option data for {ticker}...")

    # Get available expiration dates
    expirations = get_available_expirations(ticker)
    print(f"\nAvailable expiration dates: {expirations[:5]}...")

    # Download option data
    data = download_option_data(ticker, option_type='both')

    print(f"\nDownloaded {len(data)} option contracts")
    print(f"\nSample data:")
    print(data[['optionType', 'strike', 'lastPrice', 'bid', 'ask',
                'volume', 'openInterest', 'impliedVolatility']].head(10))

    # Save to file
    filename = save_option_data(data)
    print(f"\nData saved to: {filename}")