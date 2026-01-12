# Yahoo Finance Rate Limiting Guide

## 问题背景

Yahoo Finance API 经常返回以下错误：
```
yfinance.exceptions.YFRateLimitError: Too Many Requests. Rate limited. Try after a while.
```

即使请求不频繁，也可能触发限流。

## 解决方案

`model_calibration.py` 现已内置 **请求频率控制** 和 **自动重试机制**。

### 核心功能

#### 1. 速率限制器 (RateLimiter)

自动在每个 API 请求之间添加延迟：

```python
class RateLimiter:
    """确保请求之间至少间隔指定时间"""
    def __init__(self, min_interval: float = 2.0):
        self.min_interval = min_interval  # 默认 2 秒
```

**工作原理：**
- 记录上次请求时间
- 如果距离上次请求不足 2 秒，自动等待
- 在控制台显示等待进度

**示例输出：**
```
✓ Fetching data from Yahoo Finance for ^GSPC...
⏱ Rate limiting: waiting 1.5s before next request...
```

#### 2. 指数退避重试 (Exponential Backoff)

自动重试失败的请求，每次重试延迟时间加倍：

```python
def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 5.0):
    """
    重试逻辑：
    - 第 1 次失败：等待 5 秒
    - 第 2 次失败：等待 10 秒
    - 第 3 次失败：等待 20 秒
    - 仍失败：抛出异常
    """
```

**工作原理：**
- 自动识别限流错误（"rate limit", "too many requests", "429"）
- 最多重试 3 次
- 每次重试延迟时间翻倍（5s → 10s → 20s）
- 只对限流错误重试，其他错误立即失败

**示例输出：**
```
⚠ Rate limit error (attempt 1/3)
⏱ Waiting 5s before retry...
⚠ Rate limit error (attempt 2/3)
⏱ Waiting 10s before retry...
✓ Fetching data from Yahoo Finance for ^GSPC...
  Spot Price: 5,900.00
```

## 使用方法

### 基本使用（使用默认设置）

```python
from model_calibration import calibrate_models

# 默认：2 秒间隔，3 次重试
calibrated = calibrate_models(
    equity_ticker="^GSPC",
    lookback_days=756
)
```

### 自定义延迟时间

如果仍然遇到限流，增加延迟时间：

```python
# 更保守：5 秒间隔
calibrated = calibrate_models(
    equity_ticker="^GSPC",
    lookback_days=756,
    rate_limit_delay=5.0  # 增加到 5 秒
)
```

### 高频使用场景

如果需要多次运行校准（比如在循环中）：

```python
import time

results = []
for ticker in ["^GSPC", "^DJI", "^IXIC"]:
    # 每次循环之间额外等待
    time.sleep(10)  # 10 秒缓冲时间

    calibrated = calibrate_models(
        equity_ticker=ticker,
        lookback_days=756,
        rate_limit_delay=5.0  # 使用更长的延迟
    )
    results.append(calibrated)
```

## 代码改进总结

### 移除的内容

✅ **Alpha Vantage 方法** - 简化代码，只保留 Yahoo Finance
```python
# 已移除
EquityMarketData.from_alpha_vantage()
```

### 新增的内容

✅ **RateLimiter 类** - 全局速率限制器
```python
_rate_limiter = RateLimiter(min_interval=2.0)
```

✅ **retry_with_backoff 函数** - 智能重试机制
```python
hist_data = retry_with_backoff(fetch_data, max_retries=3, initial_delay=5.0)
```

✅ **可配置的延迟** - calibrate_models() 支持自定义延迟
```python
calibrate_models(rate_limit_delay=5.0)  # 自定义延迟时间
```

### 应用范围

速率限制和重试机制应用于：
- ✅ Yahoo Finance 股票数据获取
- ✅ FRED 利率数据获取
- ✅ 所有 API 调用

## 最佳实践

### 1. 生产环境推荐配置

```python
# 保守设置，适合生产环境
calibrated = calibrate_models(
    equity_ticker="^GSPC",
    lookback_days=756,
    rate_limit_delay=3.0  # 3 秒间隔，更可靠
)
```

### 2. 开发环境快速测试

```python
# 最小延迟，适合开发测试（可能遇到限流）
calibrated = calibrate_models(
    equity_ticker="^GSPC",
    lookback_days=252,  # 只获取 1 年数据
    rate_limit_delay=1.0  # 最小 1 秒延迟
)
```

### 3. 缓存结果避免重复请求

```python
import pickle
from pathlib import Path

cache_file = Path("calibration_cache.pkl")

# 检查缓存
if cache_file.exists():
    print("Loading from cache...")
    with open(cache_file, 'rb') as f:
        calibrated = pickle.load(f)
else:
    print("Fetching fresh data...")
    calibrated = calibrate_models(
        equity_ticker="^GSPC",
        lookback_days=756,
        rate_limit_delay=2.0
    )

    # 保存到缓存
    with open(cache_file, 'wb') as f:
        pickle.dump(calibrated, f)
```

### 4. 定时任务（每日校准）

```python
from datetime import datetime

def daily_calibration():
    """每日运行一次，避免频繁请求"""
    print(f"Running daily calibration at {datetime.now()}")

    calibrated = calibrate_models(
        equity_ticker="^GSPC",
        lookback_days=756,
        rate_limit_delay=3.0
    )

    # 保存结果供全天使用
    save_to_database(calibrated)

    return calibrated

# 使用 cron 或任务调度器每天运行一次
# 例如：每天早上 6:00 运行
```

## 故障排除

### 问题 1：仍然遇到限流错误

**解决方案：**
```python
# 1. 增加延迟时间
calibrated = calibrate_models(rate_limit_delay=5.0)

# 2. 减少数据量
calibrated = calibrate_models(lookback_days=252)  # 只要 1 年数据

# 3. 在请求之间添加额外等待
import time
time.sleep(30)  # 等待 30 秒
calibrated = calibrate_models()
```

### 问题 2：网络超时

**解决方案：**
```python
# yfinance 的超时设置需要在环境变量中配置
import os
os.environ['YF_TIMEOUT'] = '30'  # 30 秒超时

from model_calibration import calibrate_models
calibrated = calibrate_models()
```

### 问题 3：代理或 VPN 问题

某些地区可能需要代理：
```python
import yfinance as yf

# 设置代理
proxies = {
    'http': 'http://your-proxy:port',
    'https': 'https://your-proxy:port'
}

# yfinance 会自动使用系统代理，或通过 session 配置
```

## 性能指标

### 预期执行时间

使用默认设置（rate_limit_delay=2.0）：

| 操作 | 预计时间 |
|-----|---------|
| 获取股票历史数据 | 2-5 秒 |
| 获取 8 个利率点 | 16-20 秒 (8 × 2秒) |
| 校准 Heston 模型 | <1 秒 |
| 校准 Hull-White | <1 秒 |
| **总计** | **约 20-30 秒** |

### 遇到限流时的重试时间

| 重试次数 | 额外等待时间 | 累计等待 |
|---------|------------|---------|
| 首次请求 | 0 秒 | 0 秒 |
| 第 1 次重试 | 5 秒 | 5 秒 |
| 第 2 次重试 | 10 秒 | 15 秒 |
| 第 3 次重试 | 20 秒 | 35 秒 |

## 技术细节

### 速率限制算法

```python
def wait(self):
    """等待直到满足最小间隔"""
    current_time = time.time()
    time_since_last = current_time - self.last_request_time

    if time_since_last < self.min_interval:
        wait_time = self.min_interval - time_since_last
        time.sleep(wait_time)

    self.last_request_time = time.time()
```

### 重试判断逻辑

```python
# 识别限流错误
error_str = str(e).lower()
if 'rate limit' in error_str or 'too many requests' in error_str or '429' in error_str:
    # 执行重试
    time.sleep(delay)
    delay *= 2  # 指数退避
else:
    # 非限流错误，立即失败
    raise
```

## 总结

✅ **自动处理限流** - 无需手动干预
✅ **智能重试** - 临时错误自动恢复
✅ **可配置延迟** - 灵活适应不同场景
✅ **清晰的日志** - 实时显示等待和重试状态
✅ **简化的 API** - 只保留 Yahoo Finance，更易维护

现在您可以放心使用 `model_calibration.py`，无需担心频繁的限流错误！🚀
