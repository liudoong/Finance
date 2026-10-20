"""
Step 1 of the PFE pipeline: GARCH(1,1) calibration for several underlyings.

Model for each asset's daily return r_t:
    r_t   = mu + eps_t
    eps_t = sqrt(h_t) * z_t
    h_t   = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

z_t is i.i.d. with mean 0 and variance 1:
    dist="normal": z ~ N(0,1)
    dist="t":      z ~ standardized Student-t with nu > 2 degrees of freedom
                   (scaled so Var(z) = 1, so h_t is still the conditional variance)

Return definition by asset type:
    "equity": log return     ln(P_t / P_{t-1})
    "comm"  : log return     ln(F_t / F_{t-1})   (use a roll-adjusted or constant-tenor series)
    "rates" : first difference  R_t - R_{t-1}     (same unit as the input: %, decimal or bp)

The input DataFrames can be in any date order (e.g. newest first);
they are sorted ascending and aligned on common dates before returns are taken,
so the standardized residuals of all assets share the same dates and can be
bootstrapped jointly by row in the next step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

ASSET_TYPES = ("equity", "rates", "comm")
PARAM_NAMES = ("mu", "omega", "alpha", "beta", "nu")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def to_level_series(df: pd.DataFrame | pd.Series,
                    value_col: str | None = None,
                    date_col: str | None = None) -> pd.Series:
    """Turn a price/rate table into an ascending, date-indexed Series of levels.

    Accepts either a DatetimeIndex, or a date column. If value_col / date_col
    are not given, the first datetime-like column is used as the date and the
    first remaining numeric column as the value.
    """
    if isinstance(df, pd.Series):
        s = df.copy()
    else:
        d = df.copy()
        if date_col is None and not isinstance(d.index, pd.DatetimeIndex):
            for c in d.columns:
                if pd.api.types.is_datetime64_any_dtype(d[c]):
                    date_col = c
                    break
            if date_col is None:  # try parsing the first column as dates
                first = d.columns[0]
                parsed = pd.to_datetime(d[first], errors="coerce")
                if parsed.notna().mean() > 0.9:
                    d[first] = parsed
                    date_col = first
        if date_col is not None:
            d[date_col] = pd.to_datetime(d[date_col])
            d = d.set_index(date_col)
        if value_col is None:
            numeric = d.select_dtypes(include="number").columns
            if len(numeric) == 0:
                raise ValueError("No numeric column found for the price/rate level.")
            value_col = numeric[0]
        s = d[value_col]

    s.index = pd.to_datetime(s.index)
    s = s.astype(float).dropna()
    s = s[~s.index.duplicated(keep="first")].sort_index()  # oldest -> newest
    return s


def compute_returns(levels: pd.Series, asset_type: str) -> pd.Series:
    """Daily returns according to asset type (see module docstring)."""
    if asset_type not in ASSET_TYPES:
        raise ValueError(f"asset_type must be one of {ASSET_TYPES}, got {asset_type!r}")
    if asset_type in ("equity", "comm"):
        if (levels <= 0).any():
            raise ValueError(f"{asset_type} levels must be positive for log returns.")
        r = np.log(levels).diff()
    else:  # rates
        r = levels.diff()
    return r.dropna()


# ---------------------------------------------------------------------------
# GARCH(1,1) likelihood
# ---------------------------------------------------------------------------
def garch_variance(eps: np.ndarray, omega: float, alpha: float, beta: float,
                   h0: float) -> np.ndarray:
    """Conditional variance path h_1..h_n given residuals eps_1..eps_n."""
    n = len(eps)
    h = np.empty(n)
    h[0] = h0
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
    return h


def _loglik_terms(z2: np.ndarray, h: np.ndarray, dist: str, nu: float) -> np.ndarray:
    """Per-observation log-likelihood, with z2 = eps^2 / h."""
    if dist == "normal":
        return -0.5 * (np.log(2 * np.pi) + np.log(h) + z2)
    # standardized Student-t (unit variance)
    c = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
         - 0.5 * np.log(np.pi * (nu - 2)))
    return c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2))


def _neg_loglik(theta: np.ndarray, r: np.ndarray, dist: str, h0: float) -> float:
    mu, omega, alpha, beta = theta[:4]
    nu = theta[4] if dist == "t" else np.inf
    eps = r - mu
    h = garch_variance(eps, omega, alpha, beta, h0)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return 1e10
    ll = _loglik_terms(eps ** 2 / h, h, dist, nu).sum()
    return -ll if np.isfinite(ll) else 1e10


def _numerical_hessian(f, x: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    k = len(x)
    step = rel_step * np.maximum(np.abs(x), 1e-3)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = step[i]
            ej = np.zeros(k); ej[j] = step[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej)
                       - f(x - ei + ej) + f(x - ei - ej)) / (4 * step[i] * step[j])
            H[j, i] = H[i, j]
    return H


# ---------------------------------------------------------------------------
# Single-asset fit
# ---------------------------------------------------------------------------
def fit_garch(returns: pd.Series, dist: str = "t", demean: bool = True,
              max_persistence: float = 0.9999) -> dict:
    """Fit GARCH(1,1) by maximum likelihood to one return series.

    Returns are internally rescaled to unit variance for numerical stability;
    reported parameters are in the original return units.
    """
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    r_raw = returns.to_numpy(dtype=float)
    scale = r_raw.std(ddof=1)
    r = r_raw / scale                     # unit-variance working series
    h0 = r.var(ddof=1)                    # initial variance (backcast = sample var)

    x0 = [r.mean() if demean else 0.0, 0.05 * h0, 0.05, 0.90]
    bounds = [(-1.0, 1.0) if demean else (0.0, 0.0),
              (1e-8, 10.0), (0.0, 1.0), (0.0, 1.0)]
    if dist == "t":
        x0.append(8.0)
        bounds.append((2.05, 200.0))
    cons = [{"type": "ineq", "fun": lambda th: max_persistence - th[2] - th[3]}]

    res = minimize(_neg_loglik, np.array(x0), args=(r, dist, h0),
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10})
    theta = res.x

    # standard errors from the inverse Hessian (in working units)
    try:
        free = [i for i, (lo, hi) in enumerate(bounds) if lo != hi]
        def f_free(xf):
            th = theta.copy(); th[free] = xf
            return _neg_loglik(th, r, dist, h0)
        cov = np.linalg.inv(_numerical_hessian(f_free, theta[free]))
        se_w = np.full(len(theta), np.nan)
        se_w[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se_w = np.full(len(theta), np.nan)

    # back to original units: mu*s, omega*s^2; alpha, beta, nu unchanged
    unit = np.array([scale, scale ** 2, 1.0, 1.0, 1.0])[:len(theta)]
    p = dict(zip(PARAM_NAMES, theta * unit))
    se = dict(zip(PARAM_NAMES, se_w * unit))
    if dist == "normal":
        p["nu"] = np.inf
        se["nu"] = np.nan

    eps = r_raw - p["mu"]
    h = garch_variance(eps, p["omega"], p["alpha"], p["beta"], h0 * scale ** 2)
    z = eps / np.sqrt(h)
    persistence = p["alpha"] + p["beta"]
    long_run_var = p["omega"] / (1 - persistence)

    return {
        "params": p,
        "std_errors": se,
        "dist": dist,
        "persistence": persistence,
        "half_life_days": np.log(0.5) / np.log(persistence),
        "long_run_var": long_run_var,
        # one-step-ahead variance for the day after the last observation:
        # this is the starting state h_{T+1} for the projection step
        "h_next": p["omega"] + p["alpha"] * eps[-1] ** 2 + p["beta"] * h[-1],
        "cond_var": pd.Series(h, index=returns.index, name="cond_var"),
        "std_resid": pd.Series(z, index=returns.index, name="std_resid"),
        "loglik": -res.fun - len(r) * np.log(scale),   # Jacobian of rescaling
        "n_obs": len(r),
        "converged": bool(res.success),
        "message": res.message,
    }


# ---------------------------------------------------------------------------
# Multi-asset entry point
# ---------------------------------------------------------------------------
def calibrate_garch(assets: dict[str, tuple], dist: str = "t",
                    align: str = "inner", demean: bool = True,
                    ann_factor: int = 252) -> dict:
    """Calibrate GARCH(1,1) for several underlyings.

    Parameters
    ----------
    assets : {name: (df, asset_type)} or {name: (df, asset_type, value_col)}
             asset_type in {"equity", "rates", "comm"}.
             df: price/rate history in any date order (newest-first is fine).
    dist   : "t" (Student-t innovations) or "normal".
    align  : "inner" keeps only dates present for every asset, so standardized
             residuals line up row by row for the joint bootstrap.
             "none" fits each asset on its own full history.
    ann_factor : trading days per year, used only for reported annual vols.

    Returns
    -------
    dict with
      "assets"      : {name: per-asset result (see fit_garch) + type, last_level, last_date}
      "std_resid"   : DataFrame of standardized residuals, one column per asset
      "resid_corr"  : correlation matrix of standardized residuals
      "summary"     : one-row-per-asset parameter table
    """
    levels = {}
    types = {}
    for name, spec in assets.items():
        df, a_type = spec[0], spec[1]
        value_col = spec[2] if len(spec) > 2 else None
        if a_type not in ASSET_TYPES:
            raise ValueError(f"{name}: asset_type must be one of {ASSET_TYPES}")
        levels[name] = to_level_series(df, value_col=value_col)
        types[name] = a_type

    if align == "inner":
        common = None
        for s in levels.values():
            common = s.index if common is None else common.intersection(s.index)
        levels = {k: v.loc[common] for k, v in levels.items()}
    elif align != "none":
        raise ValueError("align must be 'inner' or 'none'")

    results = {}
    for name, lv in levels.items():
        r = compute_returns(lv, types[name])
        fit = fit_garch(r, dist=dist, demean=demean)
        fit["type"] = types[name]
        fit["last_level"] = float(lv.iloc[-1])
        fit["last_date"] = lv.index[-1]
        fit["returns"] = r
        results[name] = fit

    std_resid = pd.DataFrame({k: v["std_resid"] for k, v in results.items()})
    if align == "inner":
        std_resid = std_resid.dropna()

    rows = []
    for name, f in results.items():
        p = f["params"]
        rows.append({
            "asset": name, "type": f["type"], "n_obs": f["n_obs"],
            "mu": p["mu"], "omega": p["omega"], "alpha": p["alpha"],
            "beta": p["beta"], "nu": p["nu"],
            "alpha+beta": f["persistence"],
            "half_life_d": f["half_life_days"],
            "longrun_vol_ann": np.sqrt(f["long_run_var"] * ann_factor),
            "current_vol_ann": np.sqrt(f["h_next"] * ann_factor),
            "last_level": f["last_level"], "converged": f["converged"],
        })

    return {
        "assets": results,
        "std_resid": std_resid,
        "resid_corr": std_resid.corr(),
        "summary": pd.DataFrame(rows).set_index("asset"),
        "dist": dist,
        "ann_factor": ann_factor,
    }


if __name__ == "__main__":
    # Minimal usage example with synthetic, newest-first data
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(end="2026-10-16", periods=1260)
    eq = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(dates))))
    rt = 4 + np.cumsum(0.05 * rng.standard_normal(len(dates)))
    df_eq = pd.DataFrame({"date": dates, "price": eq}).iloc[::-1]
    df_rt = pd.DataFrame({"date": dates, "rate": rt}).iloc[::-1]

    out = calibrate_garch({"SPX": (df_eq, "equity"), "UST10Y": (df_rt, "rates")},
                          dist="t")
    pd.set_option("display.width", 200)
    print(out["summary"].round(4).T)
    print(out["resid_corr"].round(3))
