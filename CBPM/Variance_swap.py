import numpy as np

# Parameters
S0              = 4000 # initial price
implied_vol     = 0.2 # implied vol
strike_variance = implied_vol**2
T               = 1 # time to contract maturity in years

n_simulations   = 10000 # number of simulation path
n_days          = 252 # total trading days per year
dt              = 1 / n_days # step length per day
notional        = 1_000_000 # motional per variance unit

# Simulate price path
np.random.seed(42)
price_paths = np.zeros((n_simulations, n_days +1))
price_paths[:, 0] = S0

for t in range(1, n_days + 1):
    z = np.random.normal(size = n_simulations)
    price_paths[:, t] = price_paths[:, t-1] * np.exp(
        (0 - 0.5 * implied_vol**2) * dt + implied_vol * np.sqrt(dt) * z
    )

# Calculate Realized Variance
log_returns = np.log(price_paths[:, 1:] / price_paths[:, :-1])
realized_variances = (252 / n_days) * np.sum(log_returns**2, axis=1)

# Compute Payoff
payoffs_long  = notional * (realized_variances - strike_variance)
payoffs_short = -payoffs_long

# Compute PFE
pfe_99_long  = np.percentile(payoffs_long, 99)
pfe_99_short = -np.percentile(payoffs_short, 1) # negative direction tail risk

print(f"99% PFE (Long Variance Swap): ${pfe_99_long:,.2f}")
print(f"99% PFE (Short Variance Swap): ${pfe_99_short:,.2f}")