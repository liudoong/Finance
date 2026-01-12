import pandas as pd


def extract_options_data(excel_file_path):
    """
    Extract and organize options data from Yahoo Finance Excel file.

    Args:
        excel_file_path: Path to the Excel file with 'option' and 'price' sheets

    Returns:
        DataFrame with columns:
            - underlying_ticker: Underlying asset ticker
            - trading_date: Trading date
            - maturity_date: Option expiration date
            - strike: Strike price
            - option_type: 'CALL' or 'PUT'
            - option_contract: Original option contract name
            - option_price: Option price
            - spot_price: Underlying spot price on trading date
    """

    # Read option data
    option_df = pd.read_excel(excel_file_path, sheet_name='option')

    # Read price data and create date-to-price mapping
    price_df = pd.read_excel(excel_file_path, sheet_name='price')
    spot_price_map = dict(zip(
        pd.to_datetime(price_df['yf-1m2i7s2']).dt.date,
        price_df['yf-1m2i7s2 5']
    ))

    # Extract relevant columns from option sheet
    data = {
        'option_contract': option_df['subtle-link'],
        'trading_date_raw': option_df['yf-1oeiges'],
        'option_price': option_df['yf-1oeiges 2'],
        'strike': option_df['subtle-link 2']
    }

    df = pd.DataFrame(data)

    # Parse option contract to extract ticker, maturity, and type
    contract_parsed = df['option_contract'].str.extract(
        r'([A-Z]{1,5})W?(\d{6})([CP])(\d{8})'
    )

    df['underlying_ticker'] = contract_parsed[0]
    df['option_type'] = contract_parsed[2].map({'C': 'CALL', 'P': 'PUT'})

    # Parse maturity date (format: YYMMDD)
    maturity_str = contract_parsed[1]
    df['maturity_date'] = pd.to_datetime(
        '20' + maturity_str.str[:2] + maturity_str.str[2:4] + maturity_str.str[4:6],
        format='%Y%m%d',
        errors='coerce'
    )

    # Parse trading date (format: "M/D/YYYY H:MM AM/PM")
    df['trading_date'] = df['trading_date_raw'].str.extract(r'(\d{1,2}/\d{1,2}/\d{4})')[0]
    df['trading_date'] = pd.to_datetime(df['trading_date'], format='%m/%d/%Y', errors='coerce')

    # Map spot prices
    df['spot_price'] = df['trading_date'].dt.date.map(spot_price_map)

    # Convert dates to date objects (remove time component)
    df['trading_date'] = df['trading_date'].dt.date
    df['maturity_date'] = df['maturity_date'].dt.date

    # Select and order final columns
    result = df[[
        'underlying_ticker',
        'trading_date',
        'maturity_date',
        'strike',
        'option_type',
        'option_contract',
        'option_price',
        'spot_price'
    ]].copy()

    # Remove rows with missing critical data
    result = result.dropna(subset=['underlying_ticker', 'maturity_date'])

    # Sort by ticker, maturity, strike, and type
    result = result.sort_values([
        'underlying_ticker',
        'maturity_date',
        'strike',
        'option_type'
    ]).reset_index(drop=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Extraction Summary")
    print(f"{'='*60}")
    print(f"Total records: {len(result)}")
    print(f"Trading dates: {result['trading_date'].notna().sum()} ({result['trading_date'].notna().sum()/len(result)*100:.1f}%)")
    print(f"Spot prices: {result['spot_price'].notna().sum()} ({result['spot_price'].notna().sum()/len(result)*100:.1f}%)")
    print(f"\nOption types:\n{result['option_type'].value_counts()}")
    print(f"\nUnderlying tickers:\n{result['underlying_ticker'].value_counts()}")

    if result['spot_price'].notna().any():
        print(f"\nSpot price range: ${result['spot_price'].min():.2f} - ${result['spot_price'].max():.2f}")

    return result


# Example usage
if __name__ == "__main__":
    # Process the file
    input_file = "spx_infvol_20260109.xlsx"

    print(f"Processing {input_file}...")
    df = extract_options_data(input_file)

    # Display sample
    print(f"\n{'='*60}")
    print(f"Sample Data (first 10 rows)")
    print(f"{'='*60}")
    print(df[['underlying_ticker', 'trading_date', 'maturity_date',
              'strike', 'option_type', 'option_price', 'spot_price']].head(10))

    # Save to CSV
    output_file = input_file.replace('.xlsx', '_extracted.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
