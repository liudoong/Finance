import numpy as np
import pandas as pd

from Hist_Bs_Func import gbm_para, gbm_sim


# Step 0) Load historical data (latest date at top row).
# Replace this with your own source if needed.
try:
    hist_df = pd.read_csv(
        "Historical_Bootstrap/historical_input.csv",
        index_col=0,
        parse_dates=True,
    )
except FileNotFoundError:
    # Fallback demo data (2 assets, latest on top).
    demo_dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=800)[::-1]
    rng = np.random.default_rng(123)
    x1 = 5000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(demo_dates))))
    x2 = 2700.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.010, len(demo_dates))))
    hist_df = pd.DataFrame({"series_1": x1, "series_2": x2}, index=demo_dates)

hist_df = hist_df.iloc[:, :2].copy()
hist_df.columns = ["series_1", "series_2"]


# Step 1) Calibrate GBM parameters from historical data.
params = gbm_para(
    df=hist_df,
    start_date="2026-03-02",
    maturity_date="2028-12-29",
    switch=2,
)


# Step 2) Generate 1000 simulated future paths.
n_paths = 1000
path_s1_list = []
path_s2_list = []
date_index = None

for i in range(n_paths):
    params_i = params.copy()
    params_i["seed"] = i + 1
    sim_df = gbm_sim(params_i, switch=2)  # index is latest -> oldest

    if date_index is None:
        date_index = sim_df.index

    path_s1_list.append(sim_df["series_1"].to_numpy())
    path_s2_list.append(sim_df["series_2"].to_numpy())


# Step 3) Assemble (1000 x T) DataFrames: rows=path, cols=date.
cols = [d.strftime("%Y-%m-%d") for d in date_index]
idx = [f"path_{i:04d}" for i in range(1, n_paths + 1)]

paths_s1 = pd.DataFrame(np.vstack(path_s1_list), index=idx, columns=cols)
paths_s2 = pd.DataFrame(np.vstack(path_s2_list), index=idx, columns=cols)


# Step 4) Optional: save outputs.
paths_s1.to_csv("Historical_Bootstrap/paths_s1_1000xT.csv")
paths_s2.to_csv("Historical_Bootstrap/paths_s2_1000xT.csv")


# Step 5) Quick checks for shape and orientation.
print("paths_s1 shape:", paths_s1.shape)
print("paths_s2 shape:", paths_s2.shape)
print("first date column (latest):", paths_s1.columns[0])
print("last date column (oldest):", paths_s1.columns[-1])
