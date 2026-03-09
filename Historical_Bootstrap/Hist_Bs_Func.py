import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal, norm


def log_return(df, n):
    df = np.log(df/df.shift(-n))
    df = df.dropna()
    return df

def gbm_para(df:            pd.DataFrame,
             start_date:    str = "2025-12-31",
             maturity_date: str = "2028-12-31",
             switch:        int = 2,
             ) -> dict:
    """
    Calibrate parameters used in GBM simulation;
    if input dataframe has two columns (two sequences), switch == 2;
    if input dataframe has three columns, switch ==3.
    
    Notice: if undelrying is FX, the drift should be manually replaced by r_d - r_f
    
    return: Dict (used as input of gbm_sim() function)
    """
    
    log_ret = np.log(df / df.shift(-1)).dropna()
    mu      = log_ret.mean()
    sigma   = log_ret.std()
    
    if switch == 2:

        
        paras = {
                 "s1_start":        df.iloc[0,0],
                 "s2_start":        df.iloc[0,1],
                 "s1_annual_vol":   sigma.iloc[0] * np.sqrt(252),
                 "s2_annual_vol":   sigma.iloc[1] * np.sqrt(252),
                 "s1_annual_drift": mu.iloc[0] * 252,
                 "s2_annual_drift": mu.iloc[1] * 252,
                 "correlation":     log_ret.iloc[:,0].corr(log_ret.iloc[:,1]),
                 "start_date":      start_date,
                 "maturity_date":   maturity_date,
                 "seed":            None,  
                }
        
    elif switch == 3:
        
        paras = {
                 "s1_start":        df.iloc[0,0],
                 "s2_start":        df.iloc[0,1],
                 "s3_start":        df.iloc[0,2],
                 "s1_annual_vol":   sigma.iloc[0] * np.sqrt(252),
                 "s2_annual_vol":   sigma.iloc[1] * np.sqrt(252),
                 "s3_annual_vol":   sigma.iloc[2] * np.sqrt(252),
                 "s1_annual_drift": mu.iloc[0] * 252,
                 "s2_annual_drift": mu.iloc[1] * 252,
                 "s3_annual_drift": mu.iloc[2] * 252,
                 "rho12":           log_ret.iloc[:,0].corr(log_ret.iloc[:,1]),
                 "rho13":           log_ret.iloc[:,0].corr(log_ret.iloc[:,2]),
                 "rho23":           log_ret.iloc[:,1].corr(log_ret.iloc[:,2]),
                 "start_date":      start_date,
                 "maturity_date":   maturity_date,
                 "seed":            None,  
                }
        
    return paras


def gbm_sim(params: dict,
            switch: int = 2,
            ) -> pd.DataFrame:
    
    """
    Generate GBM time series
    
    Return：DataFrame，Column ['series1', 'series2']，useing date as index 
    """
    
    if switch == 2:
    
        defaults = {
                   "s1_start":        5000.0,        # Start value of sequence
                   "s2_start":        2700.0,        
                   "s1_annual_vol":   0.18,          # Annualized volatility
                   "s2_annual_vol":   0.15,
                   "s1_annual_drift": 0.08,          # Annualized drift
                   "s2_annual_drift": 0.05,
                   "correlation":     -0.15,         # Correlation of the two sequences
                   "start_date":      "2025-12-31",  # Start date
                   "maturity_date":   "2028-12-29",  # End date
                   "seed":            None,            # Random seed
                   }
        
        p = {**defaults, **params}
    
        np.random.seed(p["seed"])
        dt = 1 / 252
    
        # derive n_days from dates
        dates     = pd.bdate_range(start=p["start_date"], end=p["maturity_date"])
        n_days    = len(dates)
        if n_days == 0:
            raise ValueError("No business days between start_date and maturity_date.")
        
        # derive covariance matrix and choloskey decomposition
        cov_matrix = np.array([
                              [1.0, p["correlation"]],
                              [p["correlation"], 1.0],
                              ])
        
        L          = cholesky(cov_matrix, lower=True)
    
        # derive independent random sequences, then derive the dependent random sequences
        z_independent = np.random.standard_normal((2, n_days))
        z_correlated  = L @ z_independent # shape (2,n_days)
    
        # derive GBP path
        s1_daily_vol   = p["s1_annual_vol"] * np.sqrt(dt)
        s2_daily_vol   = p["s2_annual_vol"] * np.sqrt(dt)
        s1_daily_drift = (p["s1_annual_drift"] - 0.5 * p["s1_annual_vol"]**2) * dt
        s2_daily_drift = (p["s2_annual_drift"] - 0.5 * p["s2_annual_vol"]**2) * dt
    
        s1_log_returns = s1_daily_drift + s1_daily_vol * z_correlated[0]
        s2_log_returns = s2_daily_drift + s2_daily_vol * z_correlated[1]
    
        # accumulate prices
        s1_prices = p["s1_start"] * np.exp(np.cumsum(s1_log_returns))
        s2_prices = p["s2_start"] * np.exp(np.cumsum(s2_log_returns))
    
        df = pd.DataFrame({"series_1": s1_prices, "series_2": s2_prices}, index=dates)
    
    
    elif switch == 3:
        
        defaults = {
                    "s1_start":        5000.0,
                    "s2_start":        2700.0,
                    "s3_start":        120.0,
                    "s1_annual_vol":   0.18,
                    "s2_annual_vol":   0.15,
                    "s3_annual_vol":   0.22,
                    "s1_annual_drift": 0.08,
                    "s2_annual_drift": 0.05,
                    "s3_annual_drift": 0.06,
                    "rho12":           -0.15,
                    "rho13":           0.25,
                    "rho23":           0.10,
                    "start_date":      "2025-12-31",
                    "maturity_date":   "2028-12-29",
                    "seed":            None,
                    }
        
        p        = {**defaults, **params}

        # derive n_days from dates
        dates     = pd.bdate_range(start=p["start_date"], end=p["maturity_date"])
        n_days    = len(dates)
        if n_days == 0:
            raise ValueError("No business days between start_date and maturity_date.")

        # derive time steps
        dt = 1 / 252
        np.random.seed(p["seed"])

        # derive covariance matrix and choloskey decomposition
        corr = np.array([
                        [1.0,       p["rho12"], p["rho13"]],
                        [p["rho12"], 1.0,       p["rho23"]],
                        [p["rho13"], p["rho23"], 1.0      ],
                        ])
        L    = cholesky(corr, lower=True)

        # derive independent random sequences, then derive the dependent random sequences
        z      = np.random.standard_normal((3, n_days))
        z_corr = L @ z

        # derive GBP path
        v1 = p["s1_annual_vol"] * np.sqrt(dt)
        v2 = p["s2_annual_vol"] * np.sqrt(dt)
        v3 = p["s3_annual_vol"] * np.sqrt(dt)

        d1 = (p["s1_annual_drift"] - 0.5 * p["s1_annual_vol"]**2) * dt
        d2 = (p["s2_annual_drift"] - 0.5 * p["s2_annual_vol"]**2) * dt
        d3 = (p["s3_annual_drift"] - 0.5 * p["s3_annual_vol"]**2) * dt

        r1 = d1 + v1 * z_corr[0]
        r2 = d2 + v2 * z_corr[1]
        r3 = d3 + v3 * z_corr[2]

        # accumulate price sequences
        s1 = p["s1_start"] * np.exp(np.cumsum(r1))
        s2 = p["s2_start"] * np.exp(np.cumsum(r2))
        s3 = p["s3_start"] * np.exp(np.cumsum(r3))

        df = pd.DataFrame(
                         {"series_1": s1, "series_2": s2, "series_3": s3},
                         index=dates)
            
    return df[::-1]

    
def sim_path(gbm_sim_func,
             params:  dict,
             n_paths: int = 1000,
             ) -> dict:
    """
    Simulate many GBM paths using gbm_sim-style function.

    Inputs: gbm_sim_func: usually gbm_sim
            params: dict, typically from gbm_para(...)
            n_paths: number of paths

    Returns: a dictionary with index and dataframes 
            {
              "dates": DatetimeIndex,
              "series_1": DataFrame(n_paths x T),
              "series_2": DataFrame(n_paths x T),
              ... (series_3 if 3-asset)
            }
    """

    # Infer 2-asset / 3-asset mode directly from parameter keys.
    switch = 3 if "s3_start" in params else 2

    # Initiate three empty variables 
    collected  = None # to accumulate each simulated path
    dates      = None # to save time index (T date time values)
    asset_cols = None # to save asset series names: series_1, series_2, etc. 

    for i in range(n_paths):
        p_i         = params.copy()                    # one copy of para to avoid make changes, 
        p_i["seed"] = i + 1                            # to ensure each path use different random seed
        df_i        = gbm_sim_func(p_i, switch=switch) # generate one ith path, shape (T x N)

        if dates is None:                              # Initiate date only at the first round.
            dates      = df_i.index                    # record the date index, all following path use this one
            asset_cols = list(df_i.columns)            # record asset series names, all following path use the same
            collected  = {c: [] for c in asset_cols}   # create a empty dict to save all time series at the first run

        for c in asset_cols:
            collected[c].append(df_i[c].to_numpy())    # take current asset whole time series out, transfer to numpy array,
                                                       # and append it into the dict.

    idx  = [f"path_{i:04d}" for i in range(1, n_paths + 1)]  # construct path index e.g. "path_0001, path_0002..."
    cols = [d.strftime("%Y-%m-%d") for d in dates]           # transfer index into string for column names
    out  = {"dates": dates}                                  # save original date index
    
    for c in asset_cols:
        out[c] = pd.DataFrame(np.vstack(collected[c]), index=idx, columns=cols) # construct the dataframes in output dict

    return out


def bs_path(df:            pd.DataFrame,
            start_date:    str = "2025-12-31",
            maturity_date: str = "2028-12-31",
            n_paths:       int = 1000,
            switch:        int = 2,
            seed:          int = None,
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

    return out


def copula_mtm_dual(params: dict,
                    path_dict: dict,
                    k1: float,
                    k2: float,
                    notional: float,
                    r: float = 0.03,
                    copula_rho: float = None,
                    q1: float = 0.0,
                    q2: float = 0.0,
                    ) -> pd.DataFrame:
    """
    Price dual-asset digital MtM on each (path, date) node using Gaussian copula.

    Inputs:
        params: output dict from gbm_para
        path_dict: output dict from sim_path / bs_path
        k1, k2: strikes of series_1 and series_2
        notional: payoff notional
        r: risk-free rate
        copula_rho: copula correlation, if None use params["correlation"]
        q1, q2: carry/dividend rates

    Returns:
        MtM DataFrame with shape (n_paths, T), same index/columns as path_dict["series_1"].
    """


    rho = params.get("correlation") if copula_rho is None else copula_rho

    s1_df = path_dict["series_1"]
    s2_df = path_dict["series_2"]


    sigma1 = float(params["s1_annual_vol"])
    sigma2 = float(params["s2_annual_vol"])
    maturity = pd.Timestamp(params["maturity_date"])
    cols_dt = pd.to_datetime(s1_df.columns)

    mtm = pd.DataFrame(index=s1_df.index, columns=s1_df.columns, dtype=float)
    cov = [[1.0, rho], [rho, 1.0]]
    eps = 1e-12

    for j, col in enumerate(s1_df.columns):
        t = cols_dt[j]
        tau = (maturity - t).days / 365.0

        s1 = s1_df[col].to_numpy(dtype=float)
        s2 = s2_df[col].to_numpy(dtype=float)

        if tau <= 0:
            mtm[col] = notional * ((s1 < k1) & (s2 < k2)).astype(float)
            continue

        z1 = (np.log(k1) - (np.log(s1) + (r - q1 - 0.5 * sigma1**2) * tau)) / (sigma1 * np.sqrt(tau))
        z2 = (np.log(k2) - (np.log(s2) + (r - q2 - 0.5 * sigma2**2) * tau)) / (sigma2 * np.sqrt(tau))

        p1 = np.clip(norm.cdf(z1), eps, 1.0 - eps)
        p2 = np.clip(norm.cdf(z2), eps, 1.0 - eps)
        c1 = norm.ppf(p1)
        c2 = norm.ppf(p2)

        joint_p = np.empty(len(s1))
        for i in range(len(s1)):
            joint_p[i] = multivariate_normal.cdf([c1[i], c2[i]], mean=[0.0, 0.0], cov=cov)

        mtm[col] = notional * np.exp(-r * tau) * joint_p

    return mtm
