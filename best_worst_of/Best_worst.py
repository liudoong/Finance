#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:39:36 2026

@author: october
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.optimize import minimize


def load_price_series(ticker: str, price_file: str, sheet_name: str = "Sheet1") -> pd.Series:
    """
    Read a single ticker's price history from an Excel file into a clean,
    sorted pd.Series of Adj Close prices indexed by Date.

    - Drops dividend-annotation rows (flagged by "Dividend" in the "Open" column)
    - Keeps only Date + Adj Close
    - De-duplicates rows sharing the same Date:
        - if all duplicate rows for a date agree on Adj Close -> keep the first
        - if they disagree -> raise (ambiguous, needs manual inspection)
    - Forward-fills any remaining NaN in Adj Close with the previous day's price
    """
    raw = pd.read_excel(price_file, sheet_name=sheet_name)
    raw.columns = raw.columns.str.strip()
    raw = raw[~raw["Open"].astype(str).str.contains("Dividend", case=False, na=False)]

    raw = raw[["Date", "Adj Close"]].copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()

    prices = raw["Adj Close"].rename(ticker)

    dup_mask = prices.index.duplicated(keep=False)
    if dup_mask.any():
        dup_dates = prices.index[dup_mask].unique()
        conflicting = [dt for dt in dup_dates if prices.loc[[dt]].nunique() > 1]
        if conflicting:
            raise ValueError(
                f"[{ticker}] duplicate Date rows with DIFFERING Adj Close values "
                f"on: {list(conflicting)} -- needs manual inspection, not auto-resolved."
            )
        n_dropped = int(dup_mask.sum()) - len(dup_dates)
        print(f"[{ticker}] dropped {n_dropped} duplicate-date row(s) (same value, kept first).")
        prices = prices[~prices.index.duplicated(keep="first")]

    prices = prices.ffill()

    return prices

def combine_price_series(series_list: list[pd.Series]) -> pd.DataFrame:
    """
    Combine N price series (each already indexed by Date, named by ticker)
    into a single DataFrame on the union of all dates. Any date on which a
    given ticker has no observation (holiday mismatch, missing bar, etc.)
    is left as NaN for that column -- no forward-fill or dropping here,
    that's a separate, deliberate decision left to the caller.

    Parameters
    ----------
    series_list : list of pd.Series
        Each series should have a DatetimeIndex and a .name equal to its
        ticker (as returned by load_price_series).

    Returns
    -------
    pd.DataFrame, indexed by the sorted union of all dates across tickers,
    one column per ticker, in the order given.
    """
    if len(series_list) == 0:
        raise ValueError("series_list is empty")

    names = [s.name for s in series_list]
    if any(n is None for n in names):
        raise ValueError("every series must have a .name (ticker) set")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate ticker names in series_list: {names}")

    df = pd.concat(series_list, axis=1, join="outer").sort_index()
    return df


def garch_h_path(params, r):
    omega, alpha, beta = params
    h = np.empty(len(r))
    h[0] = np.var(r)
    for t in range(1, len(r)):
        h[t] = omega + alpha * r[t-1] ** 2 + beta * h[t-1]
    return h

def negloglik(params, r):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    h = garch_h_path(params, r)
    return 0.5 * np.sum(np.log(2 * np.pi * h) + r ** 2 / h)

def calibrate_garch(returns: np.ndarray) -> dict:
    """
    Calibrate a GARCH(1,1) via QMLE on a single return series.
    Relies on the negloglik / garch_h_path formulation
    """
    opt = minimize(
                  negloglik, [1e-6, 0.05, 0.85], args=(returns,), method="Nelder-Mead",
                  options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 50000},
                  )
    omega, alpha, beta = opt.x 
    h = garch_h_path(opt.x, returns)
    z = returns / np.sqrt(h)
    z -= z.mean()
    return dict(omega = omega, alpha = alpha, beta = beta, z = z,
                h_last = h[-1], r_last = returns[-1])
    

def simulate_to_date(
    prices_df: pd.DataFrame,
    target_date: str,
    n_paths: int = 1000,
    rng_seed: int = 2026,
    r_rate: float = 0.0362,
    ) -> dict[str, pd.DataFrame]:
    """
    Simulate correlated GARCH(1,1) paths for every column (ticker) in
    prices_df -- any N -- from the last available date up to target_date,
    returning the FULL simulated price path for every business day in
    between (not just the terminal date).

    Correlation method: static, joint-date bootstrap (see previous version's
    docstring for detail) -- one historical date index is drawn per
    (path, future day), shared across all tickers.

    Returns
    -------
    dict mapping each ticker -> DataFrame of shape (n_paths, n_days), where
    n_days = 1 (the last historical date, as S0) + n_future business days
    up to and including target_date. Columns are actual dates (DatetimeIndex),
    same convention as the single-asset MU code's S_df.
    """
    tickers = list(prices_df.columns)
    dt = 1.0 / 252

    last_date = prices_df.index.max()
    target = pd.Timestamp(target_date)
    if target <= last_date:
        raise ValueError(f"target_date {target.date()} must be after last "
                          f"available price date {last_date.date()}")
    n_future = np.busday_count(last_date.date(), target.date())

    future_dates = pd.bdate_range(start=last_date, end=target)[1:]
    assert len(future_dates) == n_future
    all_dates = pd.DatetimeIndex([last_date]).append(future_dates)

    # 1) per-ticker GARCH calibration
    log_returns = np.log(prices_df).diff()
    fits = {tk: calibrate_garch(log_returns[tk].dropna().values) for tk in tickers}

    # 2) align all tickers' standardized residuals onto their common dates
    z_df = pd.DataFrame({tk: pd.Series(fits[tk]["z"], index=log_returns[tk].dropna().index)
                          for tk in tickers}).dropna(how="any")
    z_common = z_df.values
    n_common_days = z_common.shape[0]

    # 3) joint-date bootstrap: one shared historical-date index per (path, day)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n_common_days, size=(n_paths, n_future))
    z_boot = z_common[idx]          # shape (n_paths, n_future, n_tickers)

    # 4) run each ticker's GARCH recursion, keeping every intermediate step
    sim_paths = {}
    for j, tk in enumerate(tickers):
        f = fits[tk]
        h_prev = np.full(n_paths, f["h_last"])
        shock_prev = np.full(n_paths, f["r_last"])
        logS = np.empty((n_paths, n_future))
        for t in range(n_future):
            h_t = f["omega"] + f["alpha"] * shock_prev**2 + f["beta"] * h_prev
            shock_t = np.sqrt(h_t) * z_boot[:, t, j]
            increment = (r_rate * dt - 0.5 * h_t) + shock_t
            logS[:, t] = (logS[:, t-1] if t > 0 else 0.0) + increment
            h_prev, shock_prev = h_t, shock_t

        S0 = prices_df[tk].iloc[-1]
        S_full = np.empty((n_paths, n_future + 1))
        S_full[:, 0] = S0
        S_full[:, 1:] = S0 * np.exp(logS)

        sim_paths[tk] = pd.DataFrame(S_full, columns=all_dates)
        sim_paths[tk].index.name = "path"

    return sim_paths


def best_worst_payoff(
    sim_paths: dict[str, pd.DataFrame],
    strike: float,
    notional: float,
    target_date: str,
    extremum: str = "best",       # "best" or "worst"
    option_type: str = "put",     # "call" or "put"
    seller_perspective: bool = False,
) -> pd.Series:
    """
    Compute the terminal payoff of a best-of/worst-of CALL/PUT at
    target_date, for every simulated path.

    seller_perspective controls whose P&L this represents:
      - False (default) : holder/buyer's payoff -- the classic
                           max(0, strike - extremum) / max(0, extremum - strike)
                           formula. This is what the option is worth to
                           whoever is LONG it (e.g. the client, adapt).
      - True             : seller's MTM -- the negative of the above.
                           Use this when Barclays SOLD the option (as in
                           this trade): Barclays' own MTM/replacement-cost
                           position is -(holder's payoff), and is always
                           <= 0 for a short option position. This is the
                           perspective CCR exposure should be computed
                           from for Barclays' own PFE/EE.

    Everything else unchanged from the previous version -- still works
    for any N tickers.
    """
    if extremum not in ("best", "worst"):
        raise ValueError(f"extremum must be 'best' or 'worst', got {extremum!r}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    target = pd.Timestamp(target_date)
    tickers = list(sim_paths.keys())

    ratios = pd.DataFrame({
        tk: sim_paths[tk][target] / sim_paths[tk].iloc[:, 0]
        for tk in tickers
    })

    extremum_series = ratios.max(axis=1) if extremum == "best" else ratios.min(axis=1)

    if option_type == "put":
        payoff = notional * np.maximum(0.0, strike - extremum_series)
    else:
        payoff = notional * np.maximum(0.0, extremum_series - strike)

    if seller_perspective:
        payoff = -payoff

    side = "seller" if seller_perspective else "holder"
    payoff.name = f"{extremum}_of_{option_type}_payoff_{side}"
    return payoff


def compute_mtm_matrix(
    sim_paths: dict[str, pd.DataFrame],
    payoff: pd.Series,
    strike: float,
    target_date: str,
    r_rate: float = 0.0362,
    extremum: str = "best",
    option_type: str = "put",
    ) -> pd.DataFrame:
    """
    Regression-based MTM(t) for every valuation date t up to target_date,
    for a single-observation (European) best-of/worst-of option.

    No path dependency / no early resolution here, so for every t we
    estimate MTM(t) as the fitted value of a cross-sectional regression of
    the (single, T-dated, discounted-to-t) payoff on state variables
    observable at t -- avoids nested Monte Carlo, same technique underlying
    LSM continuation-value regression (just without an exercise decision).

    Regressors X at time t:
      - each ticker's moneyness m_i(t) = S_i(t)/S0_i, and its square
      - the current best/worst value across tickers, extremum(t), and its square
      - the current intrinsic value given extremum(t) -- i.e. the payoff's
        own functional form evaluated at t, helps fit the kink at strike

    Returns
    -------
    pd.DataFrame, shape (n_paths, n_days), columns = every date present in
    sim_paths (pricing/last-historical date through target_date).
    """
    if extremum not in ("best", "worst"):
        raise ValueError(f"extremum must be 'best' or 'worst', got {extremum!r}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    tickers = list(sim_paths.keys())
    target = pd.Timestamp(target_date)
    all_dates = sim_paths[tickers[0]].columns
    n_paths = len(payoff)
    S0 = {tk: sim_paths[tk].iloc[0, 0] for tk in tickers}   # scalar per ticker

    def year_frac(d1, d2):
        return (d2 - d1).days / 365.0

    def intrinsic(extremum_t):
        return (np.maximum(0.0, strike - extremum_t) if option_type == "put"
                else np.maximum(0.0, extremum_t - strike))

    mtm_cols = {}
    for t in all_dates:
        if t >= target:
            # at/after maturity: payoff is fully realized, no more time value
            mtm_cols[t] = payoff.values
            continue

        # 1) state variables at t
        m = pd.DataFrame({tk: sim_paths[tk][t] / S0[tk] for tk in tickers})
        extremum_t = m.max(axis=1) if extremum == "best" else m.min(axis=1)
        intrinsic_t = intrinsic(extremum_t)

        X_cols = [np.ones(n_paths)]
        for tk in tickers:
            X_cols.append(m[tk].values)
            X_cols.append(m[tk].values ** 2)
        X_cols += [extremum_t.values, extremum_t.values ** 2, intrinsic_t.values]
        X = np.column_stack(X_cols)

        # 2) regress payoff (discounted from T back to t) on X(t)
        disc = np.exp(-r_rate * year_frac(t, target))
        Y = payoff.values * disc

        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        mtm_cols[t] = X @ beta

    mtm_df = pd.DataFrame(mtm_cols, index=payoff.index)
    mtm_df.index.name = "path"
    return mtm_df


def compute_exposure_profile_two_stage(
    mtm_df: pd.DataFrame,
    mpor_days: int,
    pfe_quantile: float = 0.99,
) -> dict[str, pd.Series]:
    """
    Two-stage clip exposure profile (per firm policy: still requires a
    10-business-day MPOR calculation, but clipping is applied BOTH at the
    MTM level (before the diff) AND again on the MPOR diff (after).

    Stage 1 -- level clip: E_level(t) = max(V(t), 0)
      This is the base Basel exposure definition. For a position whose
      V(t) is structurally bounded on one side (e.g. a short option,
      V(t) <= 0 always), E_level(t) collapses to 0 identically -- this is
      what removes the "noise from a position moving around within its
      always-negative region" that was inflating PFE/EE in the pure
      diff-based version.

    Stage 2 -- MPOR diff + clip: 
      diff(t) = E_level(t) - E_level(t - mpor_days)
      E_diff(t) = max(diff(t), 0)
      This captures potential further exposure INCREASE over the margin
      period of risk, applied to an already-non-negative base -- consistent
      with the standard RC + PFE-add-on structure (SA-CCR / margined IMM),
      and satisfies the firm policy requirement to compute a 10-business-
      day MPOR figure.

    Returns
    -------
    dict with keys:
      "E_level"     : the stage-1 clipped level matrix (n_paths x n_days)
      "diff"        : the stage-2 diff matrix, pre-clip (n_paths x n_days,
                       NaN in first mpor_days columns)
      "EE"          : mean across paths of the stage-2 clipped diff
      "PFE"         : pfe_quantile across paths of the stage-2 clipped diff
      "EffectiveEE" : running max of EE
      "EEPE"        : running time-weighted average of EffectiveEE
    """
    E_level = mtm_df.clip(lower=0)                       # stage 1
    diff_df = E_level.diff(periods=mpor_days, axis=1)     # MPOR diff on the clipped level
    E_diff = diff_df.clip(lower=0)                        # stage 2

    EE = E_diff.mean(axis=0).dropna().rename("EE")
    PFE = E_diff.quantile(pfe_quantile, axis=0).dropna().rename("PFE")
    EffectiveEE = EE.cummax().rename("EffectiveEE")

    dates = EffectiveEE.index
    dt_years = np.diff(dates.values).astype("timedelta64[D]").astype(float) / 365.0
    dt_years = np.insert(dt_years, 0, 0.0)
    cum_weighted = np.cumsum(EffectiveEE.values * dt_years)
    cum_time = np.cumsum(dt_years)
    cum_time[cum_time == 0] = np.nan
    EEPE = pd.Series(cum_weighted / cum_time, index=dates, name="EEPE")

    return {
        "E_level": E_level,
        "diff": diff_df,
        "EE": EE,
        "PFE": PFE,
        "EffectiveEE": EffectiveEE,
        "EEPE": EEPE,
    }


def plot_exposure_profile(
    profile: dict[str, pd.Series],
    title: str = "Exposure Profile",
) -> None:
    """
    Plot every Series in the compute_exposure_profile() output dict on one
    chart (EE, PFE, EffectiveEE, EEPE). The "diff" key (a DataFrame, not a
    Series) is skipped -- it's the raw matrix, not something you'd plot as
    a single curve.

    Parameters
    ----------
    profile : dict, output of compute_exposure_profile()
    title : chart title
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    style = {
        "PFE":          dict(color="#1f77b4", linewidth=1.5, linestyle="-"),
        "EE":           dict(color="gray",    linewidth=1.2, linestyle="--"),
        "EffectiveEE":  dict(color="#2ca02c", linewidth=1.2, linestyle="-."),
        "EEPE":         dict(color="#d62728", linewidth=1.5, linestyle=":"),
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, series in profile.items():
        if not isinstance(series, pd.Series):
            continue   # skip "diff" (the raw matrix)
        kwargs = style.get(name, dict(linewidth=1.2))
        ax.plot(series.index, series.values, label=name, **kwargs)

    ax.set_title(title)
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


n_paths        = 100_000
notional       = 10_000_000
target_date    = "2027-12-30"
risk_free_rate = 0.0354
mpor           = 10



tickers = {"CRM":  "CRM.xlsx",
           "NVDA": "NVDA.xlsx",
           "AVGO": "AVGO.xlsx",
           "ORCL": "ORCL.xlsx",
           "META": "META.xlsx"
           }


series_list = [load_price_series(ticker, file) for ticker, file in tickers.items()]
prices_df   = combine_price_series(series_list)

sim_result   = simulate_to_date(prices_df, target_date = target_date, n_paths = n_paths, rng_seed = 2026,)
sim_result_1 = {k: sim_result[k].copy() for k in ['NVDA','AVGO','CRM']}
sim_result_2 = {k: sim_result[k].copy() for k in ['NVDA','ORCL','META']}



payoff_701   = best_worst_payoff(sim_result_1, strike = 0.7, notional = notional, target_date = target_date, extremum = "best", option_type = "put", seller_perspective = False,)
payoff_702   = best_worst_payoff(sim_result_2, strike = 0.7, notional = notional, target_date = target_date, extremum = "best", option_type = "put", seller_perspective = False,)
payoff_601   = best_worst_payoff(sim_result_1, strike = 0.6, notional = notional, target_date = target_date, extremum = "best", option_type = "put", seller_perspective = False,)
payoff_602   = best_worst_payoff(sim_result_2, strike = 0.6, notional = notional, target_date = target_date, extremum = "best", option_type = "put", seller_perspective = False,)



mtm_df_701   = compute_mtm_matrix(sim_result_1, payoff_701, strike = 0.7, target_date = target_date, r_rate = risk_free_rate, extremum = "best", option_type = "put")
mtm_df_702   = compute_mtm_matrix(sim_result_2, payoff_702, strike = 0.7, target_date = target_date, r_rate = risk_free_rate, extremum = "best", option_type = "put")
mtm_df_601   = compute_mtm_matrix(sim_result_1, payoff_601, strike = 0.6, target_date = target_date, r_rate = risk_free_rate, extremum = "best", option_type = "put")
mtm_df_602   = compute_mtm_matrix(sim_result_2, payoff_602, strike = 0.6, target_date = target_date, r_rate = risk_free_rate, extremum = "best", option_type = "put")



profile_701  = compute_exposure_profile_two_stage(mtm_df_701, mpor_days=mpor)
profile_702  = compute_exposure_profile_two_stage(mtm_df_702, mpor_days=mpor)
profile_601  = compute_exposure_profile_two_stage(mtm_df_601, mpor_days=mpor)
profile_602  = compute_exposure_profile_two_stage(mtm_df_602, mpor_days=mpor)

PFE          = {'NVDA, AVGO, CRM, 70%': profile_701['PFE'].copy,
                'NVDA, ORCL, META, 70%': profile_702['PFE'].copy,
                'NVDA, AVGO, CRM, 60%': profile_601['PFE'].copy,
                'NVDA, ORCL, META, 60%': profile_602['PFE'].copy,
                }

PFE = pd.concat([profile_701['PFE'],profile_702['PFE'],profile_601['PFE'],profile_602['PFE']],axis=1)
PFE.columns = ['NVDA, AVGO, CRM, 70%','NVDA, ORCL, META, 70%','NVDA, AVGO, CRM, 60%','NVDA, ORCL, META, 60%']


PFE.plot(kind = 'line',figsize = (10, 6))
plt.title ('10 days MPoR PFE')
plt.xlabel('time')
plt.ylabel('PFE')

plt.tight_layout()
plt.show()

PFE_value = PFE.max()



