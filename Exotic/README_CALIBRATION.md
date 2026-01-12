# Model Calibration Workflow

本文档说明如何使用拆分后的模块结构进行 Heston 和 Hull-White 模型校准。

## 文件结构

```
Exotic/
├── model_calibration.py          # 模型校准模块（独立运行）
├── Barrier_option_inputs.py      # 主输入模块（使用校准后的参数）
└── README_CALIBRATION.md         # 本文档
```

## 工作流程

### 方法 1: 分步运行（推荐）

#### 第 1 步：独立运行模型校准

```python
# 运行校准模块（只需运行一次，或定期更新）
python model_calibration.py
```

这将：
- 从 Yahoo Finance 获取 SPX 的历史价格数据（3年）
- 从 FRED 获取美国国债利率曲线
- 校准 Heston 模型参数（使用历史数据）
- 设置 Hull-White 模型参数（使用标准市场参数）
- 打印完整的校准结果

**输出示例：**
```
================================================================================
MODEL CALIBRATION - HESTON & HULL-WHITE
================================================================================

STEP 1: Fetching Equity Market Data
--------------------------------------------------------------------------------
✓ Fetching data from Yahoo Finance for ^GSPC...
  Spot Price: 5,900.00
  Historical Vol: 15.23%
  Historical Prices: 756 days

STEP 2: Fetching Rates Market Data
--------------------------------------------------------------------------------
✓ Fetching USD rates from FRED (last 30 days)...
   1M:  4.50%
   3M:  4.55%
   6M:  4.60%
   ...

STEP 3: Calibrating Heston Model
--------------------------------------------------------------------------------
✓ Estimating Heston parameters (Hybrid Method):
  v0 (initial var):  0.023190  (vol=15.23%)
  θ (long-term var): 0.022500  (vol=15.00%)
  κ (mean reversion):2.1234
  σ (vol of vol):    0.2987
  ρ (correlation):   -0.7123

STEP 4: Validating Heston Parameters
--------------------------------------------------------------------------------
✓ Feller condition satisfied: 2κθ = 0.0956 > σ² = 0.0892

STEP 5: Calibrating Hull-White Model
--------------------------------------------------------------------------------
✓ Hull-White Model Parameters:
  Mean reversion (a): 0.0500
  Volatility (σ):     0.0100

================================================================================
CALIBRATION COMPLETE
================================================================================
```

#### 第 2 步：在主模块中使用校准参数

有两种方式使用校准参数：

##### 方式 A：在代码中导入校准模块

```python
from Barrier_option_inputs import create_spx_barrier_example

# 使用校准后的模型参数（会自动运行校准）
inputs = create_spx_barrier_example(use_calibrated_models=True)

# 访问校准后的参数
heston_params = inputs['heston_params']
hull_white_params = inputs['hull_white_params']
equity_data = inputs['equity_data']
rates_data = inputs['rates_data']

print(f"Heston v0: {heston_params.v0:.6f}")
print(f"Heston theta: {heston_params.theta:.6f}")
print(f"Hull-White mean reversion: {hull_white_params.mean_reversion:.4f}")
```

##### 方式 B：手动导入并保存参数

```python
from model_calibration import calibrate_models

# 运行一次校准
calibrated = calibrate_models(
    equity_ticker="^GSPC",
    use_yahoo=True,
    lookback_days=756
)

# 保存参数供后续使用
heston_params = calibrated['heston_params']
hull_white_params = calibrated['hull_white_params']

# 在您的模拟中使用这些参数
# 无需每次都重新校准
```

### 方法 2: 使用硬编码参数（快速测试）

如果不想每次都运行校准（例如在开发或测试时）：

```python
from Barrier_option_inputs import create_spx_barrier_example

# 使用硬编码的示例参数（无需联网）
inputs = create_spx_barrier_example(use_calibrated_models=False)
```

## 模块说明

### `model_calibration.py` - 模型校准模块

**功能：**
- 从市场数据源获取实时数据（Yahoo Finance, FRED）
- 校准 Heston 模型参数（基于历史价格）
- 设置 Hull-White 模型参数
- 验证模型参数（Feller 条件等）

**主要函数：**
```python
calibrate_models(
    equity_ticker="^GSPC",    # 股票代码
    use_yahoo=True,            # 使用 Yahoo Finance
    lookback_days=756          # 历史数据天数（3年）
)
```

**返回值：**
```python
{
    'equity_data': EquityMarketData(...),
    'rates_data': RatesMarketData(...),
    'heston_params': HestonModelParams(...),
    'hull_white_params': HullWhiteModelParams(...)
}
```

### `Barrier_option_inputs.py` - 主输入模块

**功能：**
- 定义产品参数（障碍期权）
- 定义市场数据结构
- 定义模型参数类
- 定义模拟参数
- **可选**：从 `model_calibration.py` 导入校准参数

**主要函数：**
```python
create_spx_barrier_example(
    use_calibrated_models=False  # True: 使用校准参数, False: 使用硬编码参数
)
```

## 使用场景

### 场景 1：开发和测试
使用硬编码参数，无需每次运行校准：
```python
inputs = create_spx_barrier_example(use_calibrated_models=False)
```

### 场景 2：生产运行
定期运行校准（例如每日），保存参数并使用：
```python
# 每日运行一次
calibrated = calibrate_models()
save_parameters(calibrated)  # 保存到文件

# 在模拟中加载保存的参数
heston_params = load_heston_parameters()
hull_white_params = load_hull_white_parameters()
```

### 场景 3：研究和分析
实时校准并立即使用：
```python
inputs = create_spx_barrier_example(use_calibrated_models=True)
# 此时参数是基于最新市场数据校准的
```

## 优势

1. **模块化**：校准逻辑与主模拟逻辑分离
2. **灵活性**：可以选择使用校准参数或硬编码参数
3. **效率**：无需每次运行都重新校准
4. **可维护性**：校准代码独立，易于更新和测试
5. **可复用性**：校准模块可用于其他项目

## 依赖包

```bash
# 基础包
pip install numpy pandas

# 市场数据获取
pip install yfinance pandas-datareader

# 模型校准（可选）
pip install scipy

# QuantLib（可选，用于更高级的功能）
pip install QuantLib-Python
```

## 示例完整工作流

```python
# ============================================================
# 示例 1: 完整的校准和模拟流程
# ============================================================

from model_calibration import calibrate_models
from Barrier_option_inputs import create_spx_barrier_example, print_input_summary

# 步骤 1: 校准模型（获取最新市场数据）
print("步骤 1: 校准模型...")
calibrated = calibrate_models(
    equity_ticker="^GSPC",
    use_yahoo=True,
    lookback_days=756
)

# 步骤 2: 创建输入配置（使用校准参数）
print("\n步骤 2: 创建输入配置...")
inputs = create_spx_barrier_example(use_calibrated_models=True)

# 步骤 3: 打印输入摘要
print("\n步骤 3: 查看输入摘要...")
print_input_summary(inputs)

# 步骤 4: 运行您的模拟
print("\n步骤 4: 运行模拟...")
# 在这里调用您的模拟代码
# results = run_barrier_option_simulation(inputs)


# ============================================================
# 示例 2: 快速测试（使用硬编码参数）
# ============================================================

from Barrier_option_inputs import create_spx_barrier_example

# 直接使用硬编码参数，无需网络连接
inputs = create_spx_barrier_example(use_calibrated_models=False)

# 立即开始测试
# test_your_code(inputs)
```

## 注意事项

1. **数据源限制**：
   - Yahoo Finance: 可能被限流，但通常稳定
   - FRED: 免费但可能有延迟

2. **校准频率**：
   - 每日校准：适用于生产环境
   - 每周/月校准：适用于研究
   - 一次性校准：适用于回测

3. **参数持久化**：
   - 建议将校准参数保存到文件（JSON, pickle 等）
   - 这样可以避免每次都重新校准

4. **Feller 条件**：
   - 校准后始终检查 Feller 条件
   - 如果违反，可能需要调整参数或使用不同的估计方法

## 后续改进建议

1. **参数持久化**：添加保存/加载校准参数到文件的功能
2. **高级校准**：使用期权市场数据校准（volatility surface）
3. **多资产**：扩展支持多个标的资产
4. **实时更新**：添加定时任务自动更新校准
5. **参数验证**：添加更多的参数合理性检查
