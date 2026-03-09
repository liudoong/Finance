import numpy as np
import pandas as pd
from Hist_Bs_Func import gbm_para, gbm_sim, sim_path, bs_path
from scipy.stats import multivariate_normal, norm



# 0) Build a tiny historical sample (latest date on top)
dates = pd.bdate_range(end="2026-02-24", periods=30)[::-1]
rng = np.random.default_rng(7)
s1 = 5000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates))))
s2 = 2700 * np.exp(np.cumsum(rng.normal(0.0001, 0.009, len(dates))))
hist_df = pd.DataFrame({"series_1": s1, "series_2": s2}, index=dates)

# 1) Calibrate params from history
params = gbm_para(
    df=hist_df,
    start_date="2026-03-02",
    maturity_date="2026-06-30",
    switch=2
)

# 2) Generate ONE GBM path (T x 2)
one_path = gbm_sim(params, switch=2)
print("one_path shape:", one_path.shape)
print(one_path.head(3))

# 3) Generate MANY paths (dict output)
many = sim_path(gbm_sim, params, n_paths=5)

# many["series_1"] and many["series_2"] are (n_paths x T)
print("series_1 shape:", many["series_1"].shape)
print("series_2 shape:", many["series_2"].shape)
print(many["series_1"].iloc[:2, :3])  # first 2 paths, first 3 dates

# 3) simulate using bootstraop
bs_many = bs_path(
    df=hist_df,
    start_date="2026-03-02",
    maturity_date="2026-06-30",
    switch=2,
    n_paths = 1000
)


series1 = many["series_1"]
series2 = bs_many["series_2"]


mtm = copula_mtm_dual(
    params=params,
    path_dict=many,
    k1=5800.0,
    k2=2550.0,
    notional=1_000_000,
    r=0.03
)