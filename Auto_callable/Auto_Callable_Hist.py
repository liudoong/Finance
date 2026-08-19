import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal, norm

#%%

# 0) Build a tiny historical sample (latest date on top)
dates = pd.bdate_range(end="2026-02-24", periods=30)[::-1]
rng = np.random.default_rng(7)
s1 = 5000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates))))
s2 = 2700 * np.exp(np.cumsum(rng.normal(0.0001, 0.009, len(dates))))
hist_df = pd.DataFrame({"series_1": s1, "series_2": s2}, index=dates)

start_date    = "2026-02-25"
maturity_date = "2026-05-30"
seed          = None
n_paths       = 100
switch        = 2

initial_levels = {"series_1": 5001.06, 
                  "series_2": 2663.33,
                  }

#%%


def log_return(df, n):
    df = np.log(df/df.shift(-n))
    df = df.dropna()
    return df

def bs_path(df:            pd.DataFrame,
            start_date:    str  = "2025-12-31",
            maturity_date: str  = "2028-12-31",
            n_paths:       int  = 1000,
            switch:        int  = 2,
            seed:          int  = None
            ) -> dict:
    """
    Bootstrap one-day log returns and reconstruct path panels.

    Input dataframe convention: latest date on top, older observations below.

    Returns: a dictionary with index and dataframes 
            {
              "dates": DatetimeIndex,
              "series_1": DataFrame(n_paths x T),
              "series_2": DataFrame(n_paths x T),
              ... (series_3 if 3-asset)
            }
    """

    # One-day log returns for descending-date data.
    hist    = df.iloc[:, :switch].copy()
    log_ret = np.log(hist / hist.shift(-1)).dropna()

    # Build date grid and bootstrap returns.
    dates_asc  = pd.bdate_range(start=start_date, end=maturity_date)
    n_days     = len(dates_asc)

    rng        = np.random.default_rng(seed)
    ret_values = log_ret.to_numpy()  # shape: (n_hist, N)
    n_hist     = ret_values.shape[0]

    # Row-wise bootstrap preserves cross-asset dependence at each time step.
    pick_idx = rng.integers(0, n_hist, size=(n_paths, n_days))
    boot_ret = ret_values[pick_idx]  # shape: (n_paths, n_days, N)

    # Reconstruct prices from latest observed levels.
    s0     = hist.iloc[0].to_numpy(dtype=float)  # latest levels
    prices = s0 * np.exp(np.cumsum(boot_ret, axis=1))  # shape: (n_paths, n_days, N)

    # Generate out put dict: latest date at first column.
    dates_desc = dates_asc[::-1] # switch date order
    cols       = [d.strftime("%Y-%m-%d") for d in dates_desc]
    idx        = [f"path_{i:04d}" for i in range(1, n_paths + 1)]

    out = {"dates": dates_desc}
    for j in range(switch):
        out[f"series_{j+1}"] = pd.DataFrame(prices[:, ::-1, j],
                                            index=idx,
                                            columns=cols,
                                            )
    #reverse the result
    for key in out:
        if key == "dates":
            out["dates"] = out["dates"][::-1]
        else:
            out[key] = out[key].iloc[:, ::-1]

    return out

def to_ratio(bs_out: dict, initial_levels: dict) -> dict:
    """
    Convert bs_path output (spot prices) to ratios relative to initial levels.
    
    bs_out         : output dict from bs_path (after reversal, oldest date first)
    initial_levels : e.g. {"series_1": 100.0, "series_2": 50.0, ...}
    
    Returns: dict with same structure as bs_out, prices replaced by ratios (e.g. 1.05 = 105%)
    """
    out = {"dates": bs_out["dates"]}
    
    for key, df in bs_out.items():
        if key == "dates":
            continue
        initial = initial_levels[key]
        out[key] = df / initial
    
    return out


#%%

usecase = bs_path(hist_df, start_date, maturity_date, n_paths, switch, seed)
usecase['series_1']

usecase_ratio = to_ratio(usecase, initial_levels)
usecase_ratio["series_1"]