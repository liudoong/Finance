# Automatic Market Data Fetching

## Overview

自动从免费数据源获取市场数据，无需手动输入。

## Features

### 1. Equity Data (Yahoo Finance)
- **数据源**: Yahoo Finance (免费)
- **数据内容**:
  - 现货价格 (Spot Price)
  - 历史波动率 (Historical Volatility) - 基于过去252天
  - 历史价格序列 (Historical Prices)

### 2. Rates Data (FRED)
- **数据源**: Federal Reserve Economic Data (免费)
- **数据内容**:
  - US Treasury 零息利率曲线
  - 期限: 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 30Y

## Installation

```bash
pip install yfinance pandas-datareader
```

## Usage

### 方法 1: 自动获取数据

```python
from Barrier_option_inputs import EquityMarketData, RatesMarketData

# 自动获取 SPX 股票数据
equity_data = EquityMarketData.from_yahoo(
    ticker="^GSPC",      # S&P 500 Index
    lookback_days=252,   # 1 year历史数据
    repo_rate=0.0        # Repo利率（可选）
)

# 自动获取 USD 利率曲线
rates_data = RatesMarketData.from_fred(
    currency="USD"       # 目前只支持USD
)
```

### 方法 2: 手动输入数据（原方法）

```python
from Barrier_option_inputs import EquityMarketData, RatesMarketData
from datetime import datetime

# 手动输入股票数据
equity_data = EquityMarketData(
    spot_price=5900.0,
    atm_volatility=0.15,
    repo_rate=0.045,
    dividend_yield=0.0,
    vol_surface=None,
    historical_prices=None
)

# 手动输入利率数据
rates_data = RatesMarketData(
    curve_date=datetime(2026, 1, 9),
    tenor_points=["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"],
    zero_rates=[0.045, 0.046, 0.045, 0.043, 0.041, 0.039, 0.038],
    day_count="ACT/360",
    calendar="UnitedStates"
)
```

## 常见 Tickers

### US Indices
- S&P 500: `^GSPC`
- Dow Jones: `^DJI`
- NASDAQ: `^IXIC`
- Russell 2000: `^RUT`

### Individual Stocks
- Apple: `AAPL`
- Microsoft: `MSFT`
- Tesla: `TSLA`

## Test Script

运行测试脚本查看效果：

```bash
python test_auto_data_fetch.py
```

## Important Notes

1. **历史波动率 vs 隐含波动率**:
   - 我们使用历史波动率（从历史价格计算）
   - 不使用隐含波动率（需要付费数据源）

2. **Rate Limiting**:
   - Yahoo Finance 有访问频率限制
   - 如遇到限速，稍后重试

3. **数据延迟**:
   - FRED 数据通常有1-2天延迟
   - Yahoo Finance 数据实时性较好

4. **数据质量**:
   - 免费数据足够用于学术研究和原型开发
   - 生产环境建议使用付费数据源（Bloomberg, Refinitiv等）

## Example Output

```
✓ Fetched data for ^GSPC:
  Spot Price: 5,900.00
  Historical Vol (1Y): 15.23%
  Historical Prices: 252 days

✓ Fetching most recent USD rates from FRED (last 30 days)
    1M:  4.32%
    3M:  4.45%
    6M:  4.38%
    1Y:  4.21%
    2Y:  4.05%
    5Y:  3.87%
   10Y:  3.92%
   30Y:  4.15%
✓ Successfully fetched 8 tenor points
```

## Heston Parameter Estimation

**新功能**: 自动从历史数据估计 Heston 模型参数！

```python
from Barrier_option_inputs import EquityMarketData, HestonModelParams

# 1. 获取历史数据
equity_data = EquityMarketData.from_yahoo('^GSPC', lookback_days=252)

# 2. 自动估计 Heston 参数
heston_params = HestonModelParams.estimate_from_historical(equity_data)

# 3. 验证 Feller 条件
heston_params.validate()
```

**估计方法**:
- `v0`: 最近3个月的历史波动率方差
- `θ`: 全样本的长期波动率方差
- `κ`: 从波动率的自回归特性估计均值回归速度
- `σ`: 从波动率的波动率估计
- `ρ`: 收益率与波动率变化的相关性（leverage effect）

**注意**: 这是基于历史数据的快速估计，不是基于期权价格的完整校准。对于学术研究和原型开发足够准确。

## Next Steps

获取数据后，直接传入 QuantLib 进行定价和风险计算：

```python
# 方法1: 自动估计 (推荐)
equity_data = EquityMarketData.from_yahoo('^GSPC')
rates_data = RatesMarketData.from_fred('USD')
heston_params = HestonModelParams.estimate_from_historical(equity_data)

# 方法2: 手动指定
equity_data = EquityMarketData(spot_price=5900, atm_volatility=0.15, ...)
rates_data = RatesMarketData(curve_date=..., tenor_points=[...], ...)
heston_params = HestonModelParams(v0=0.0225, theta=0.0225, ...)

# 将数据转换为 QuantLib 对象
calculation_date = ql.Date(9, 1, 2026)
eq_ql = equity_data.to_quantlib(calculation_date)
rates_ql = rates_data.to_quantlib(calculation_date)
heston_ql = heston_params.to_quantlib()

# 使用这些数据进行期权定价和 PFE 计算...
```

## Test Scripts

1. **测试数据获取**: `python test_auto_data_fetch.py`
2. **测试 Heston 估计**: `python test_heston_calibration.py`