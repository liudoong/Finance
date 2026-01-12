"""
Smart subsampling of option data for efficient calibration.

Reduces large datasets to optimal size (50 strikes × N maturities) while preserving
information across the moneyness and term structure dimensions.
"""

import pandas as pd
import numpy as np


def subsample_options_data(df, num_strikes=50, min_options_per_maturity=5):
    """
    Intelligently subsample options data to match calibration grid size.

    Strategy:
    - Keep all unique maturities
    - For each maturity, select ~num_strikes options distributed across moneyness
    - Prioritize ATM and liquid options

    Args:
        df: DataFrame from yahoo_option_data_cleaner.py
        num_strikes: Target number of strikes per maturity (default: 50)
        min_options_per_maturity: Minimum options to keep per maturity (default: 5)

    Returns:
        Subsampled DataFrame optimized for calibration
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Required columns
    required_cols = ['maturity_date', 'strike', 'spot_price', 'option_price']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Original data: {len(df)} options")

    # Filter out invalid data
    df_valid = df[
        (df['option_price'] > 0) &
        (df['spot_price'] > 0) &
        (df['strike'] > 0)
    ].copy()

    # Calculate moneyness
    df_valid['moneyness'] = df_valid['strike'] / df_valid['spot_price']

    # Group by maturity
    unique_maturities = df_valid['maturity_date'].unique()
    print(f"Found {len(unique_maturities)} unique maturities")

    subsampled_data = []

    for maturity in unique_maturities:
        df_mat = df_valid[df_valid['maturity_date'] == maturity].copy()

        if len(df_mat) == 0:
            continue

        # How many options to keep for this maturity
        target_count = min(num_strikes, len(df_mat))
        target_count = max(target_count, min_options_per_maturity)

        if len(df_mat) <= target_count:
            # Keep all if already small
            subsampled_data.append(df_mat)
            continue

        # Strategy: Select options distributed across moneyness spectrum
        # Prioritize:
        # 1. ATM options (moneyness near 1.0)
        # 2. Wide moneyness coverage
        # 3. Both calls and puts

        df_mat['atm_distance'] = df_mat['moneyness'].apply(lambda x: np.abs(np.log(x)))

        # Split into moneyness bins
        num_bins = min(target_count // 2, 25)  # ~2 options per bin

        df_mat['moneyness_bin'] = pd.qcut(
            df_mat['moneyness'],
            q=num_bins,
            labels=False,
            duplicates='drop'
        )

        selected = []

        # From each bin, select the option closest to bin center
        for bin_id in df_mat['moneyness_bin'].unique():
            df_bin = df_mat[df_mat['moneyness_bin'] == bin_id]

            # Prefer options closer to ATM within each bin
            df_bin_sorted = df_bin.sort_values('atm_distance')

            # Take 1-3 options per bin depending on target
            n_from_bin = max(1, target_count // num_bins)
            selected.append(df_bin_sorted.head(n_from_bin))

        df_selected = pd.concat(selected, ignore_index=True)

        # If still too many, prioritize ATM
        if len(df_selected) > target_count:
            df_selected = df_selected.sort_values('atm_distance').head(target_count)

        # If too few, add more from extremes
        if len(df_selected) < target_count:
            remaining = target_count - len(df_selected)
            df_remaining = df_mat[~df_mat.index.isin(df_selected.index)]

            # Add from far OTM on both sides
            df_remaining_sorted = df_remaining.sort_values('atm_distance', ascending=False)
            df_selected = pd.concat([df_selected, df_remaining_sorted.head(remaining)])

        subsampled_data.append(df_selected)

    # Combine all maturities
    result = pd.concat(subsampled_data, ignore_index=True)

    # Drop temporary columns
    result = result.drop(['moneyness', 'atm_distance', 'moneyness_bin'],
                         axis=1, errors='ignore')

    # Sort for clean output
    result = result.sort_values(['maturity_date', 'strike']).reset_index(drop=True)

    print(f"\nSubsampling summary:")
    print(f"  Original: {len(df)} options")
    print(f"  Subsampled: {len(result)} options ({len(result)/len(df)*100:.1f}%)")
    print(f"  Maturities: {len(unique_maturities)}")
    print(f"  Avg options per maturity: {len(result)/len(unique_maturities):.1f}")

    # Show distribution
    print(f"\nOptions per maturity:")
    maturity_counts = result['maturity_date'].value_counts().sort_index()
    for mat, count in maturity_counts.items():
        print(f"  {mat}: {count} options")

    return result


def quick_subsample(df, reduction_factor=10):
    """
    Quick and simple subsampling - just take every Nth option.

    Args:
        df: DataFrame from yahoo_option_data_cleaner.py
        reduction_factor: Keep 1 out of every N options (default: 10)

    Returns:
        Subsampled DataFrame
    """
    print(f"Quick subsample: keeping 1/{reduction_factor} of data")
    print(f"Original: {len(df)} options")

    # Ensure we have variety across maturities
    result = df.groupby('maturity_date', group_keys=False).apply(
        lambda x: x.iloc[::reduction_factor]
    ).reset_index(drop=True)

    print(f"Subsampled: {len(result)} options ({len(result)/len(df)*100:.1f}%)")

    return result


def adaptive_subsample(df, target_size=500, max_per_maturity=50):
    """
    Adaptive subsampling to achieve target total size.

    Args:
        df: DataFrame from yahoo_option_data_cleaner.py
        target_size: Target total number of options (default: 500)
        max_per_maturity: Maximum options per maturity (default: 50)

    Returns:
        Subsampled DataFrame
    """
    print(f"Adaptive subsample to ~{target_size} total options")
    print(f"Original: {len(df)} options")

    n_maturities = df['maturity_date'].nunique()
    per_maturity = min(max_per_maturity, target_size // n_maturities)

    print(f"Target: {per_maturity} options per maturity")

    return subsample_options_data(df, num_strikes=per_maturity)


# Example usage
if __name__ == "__main__":
    from yahoo_option_data_cleaner import extract_options_data

    # Load data
    print("="*70)
    print("Loading option data...")
    print("="*70)
    df = extract_options_data("spx_infvol_20260109.xlsx")

    print(f"\n\n{'='*70}")
    print("Method 1: Smart Subsampling (Recommended)")
    print("="*70)

    # Smart subsample
    df_subsampled = subsample_options_data(df, num_strikes=50)

    print(f"\n\n{'='*70}")
    print("Method 2: Quick Subsampling")
    print("="*70)

    # Quick subsample
    df_quick = quick_subsample(df, reduction_factor=10)

    print(f"\n\n{'='*70}")
    print("Method 3: Adaptive Subsampling")
    print("="*70)

    # Adaptive subsample
    df_adaptive = adaptive_subsample(df, target_size=500)

    # Save results
    df_subsampled.to_csv("options_subsampled_smart.csv", index=False)
    df_quick.to_csv("options_subsampled_quick.csv", index=False)
    df_adaptive.to_csv("options_subsampled_adaptive.csv", index=False)

    print(f"\n\n{'='*70}")
    print("Results saved to CSV files")
    print("="*70)
    print("✓ options_subsampled_smart.csv")
    print("✓ options_subsampled_quick.csv")
    print("✓ options_subsampled_adaptive.csv")

    # Test calibration speed
    print(f"\n\n{'='*70}")
    print("Testing calibration with subsampled data...")
    print("="*70)

    try:
        from vol_surface_calibrator import calibrate_vol_surface
        import time

        print("\nCalibrating SABR model...")
        start = time.time()
        result = calibrate_vol_surface(df_subsampled, model='SABR')
        elapsed = time.time() - start

        print(f"\n✓ Calibration completed in {elapsed:.1f} seconds")
        print(f"  Used {len(df_subsampled)} options (vs {len(df)} original)")
        print(f"  Speedup: {len(df)/len(df_subsampled):.1f}x fewer data points")

    except ImportError:
        print("Skipping calibration test (vol_surface_calibrator not available)")
