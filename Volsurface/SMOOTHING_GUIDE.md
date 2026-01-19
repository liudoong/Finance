# 波动率曲面平滑校准指南

## 🎯 问题背景

原始校准结果出现 "bumps"（颠簸）和过拟合问题，主要原因：

1. **数据质量问题**
   - 低流动性期权定价不准确
   - 异常值（outliers）扰乱校准
   - 深度价内/价外期权信息量低

2. **校准过拟合**
   - 模型试图精确拟合每个数据点
   - 相邻到期日参数跳跃过大
   - 没有平滑约束

3. **插值问题**
   - RBF 插值 `smoothing=0.0` 导致精确插值（过拟合）
   - 缺乏跨到期日的参数平滑

---

## 🛠️ 解决方案概览

### 方案一：改进版校准器 `smooth_calibrator.py` ⭐ **推荐**

**新增功能：**

1. **数据质量过滤** (`filter_option_data_for_smoothness`)
   - 移除时间价值不足的期权（< 5%）
   - 限制 moneyness 范围（0.7-1.5）
   - Z-score 异常值检测（默认阈值 2.5）
   - 价格合理性检查

2. **正则化校准** (`calibrate_sabr_smooth`)
   - Tikhonov 正则化：惩罚相邻到期日参数跳跃
   - 参数初始值从前一到期日继承（增强连续性）
   - 可调节正则化强度 `regularization` 参数

3. **更严格的参数边界**
   - rho: (-0.95, 0.95) 而非 (-0.99, 0.99)
   - nu: (0.05, 1.5) 而非 (0.01, 2.0)
   - 提高数值稳定性

### 方案二：修改原校准器 `vol_surface_calibrator.py`

**改进：**
- RBF 插值器 `smoothing` 从 0.0 改为 0.005
- 这会在插值阶段增加平滑度

---

## 📖 使用方法

### 方法 1：使用新的平滑校准器（推荐）

```python
from smooth_calibrator import smooth_calibrate_vol_surface
from option_data_cleaner import extract_options_data

# 加载数据
df = extract_options_data("spx_data.xlsx")

# 平滑校准 - 推荐设置
result = smooth_calibrate_vol_surface(
    df,
    model='SABR',
    regularization=0.01,        # 正则化强度（0=无平滑，0.1=强平滑）
    filter_data=True,            # 应用数据质量过滤
    outlier_threshold=2.5,       # 异常值 Z-score 阈值
    risk_free_rate=0.05,
    output_strikes=50
)

# 提取结果
surface = result['surface']
params = result['params']

# 测试插值
vol = surface.get_vol(strike=7000, maturity=1.0)
print(f"Implied Vol: {vol:.2%}")

# 绘图
surface.plot()
```

### 方法 2：使用改进的原校准器

```python
from vol_surface_calibrator import calibrate_vol_surface
from option_data_cleaner import extract_options_data

df = extract_options_data("spx_data.xlsx")

# 现在内置了 smoothing=0.005
result = calibrate_vol_surface(df, model='SABR', output_strikes=50)

surface = result['surface']
surface.plot()
```

---

## 🎛️ 参数调优建议

### 1. **Regularization（正则化强度）**

| 值 | 效果 | 适用场景 |
|---|---|---|
| 0.0 | 无正则化，可能过拟合 | 数据质量极高 |
| 0.005 | 轻度平滑 | 正常市场数据 |
| **0.01** | **中度平滑（推荐）** | **大多数情况** |
| 0.05 | 强平滑 | 噪声很大的数据 |
| 0.1 | 极强平滑，可能欠拟合 | 极少数据点 |

### 2. **Outlier Threshold（异常值阈值）**

| 值 | 效果 | 过滤严格度 |
|---|---|---|
| 3.0 | 保留更多数据 | 宽松 |
| **2.5** | **平衡（推荐）** | **中等** |
| 2.0 | 更严格过滤 | 严格 |
| 1.5 | 极严格，可能丢失有效数据 | 非常严格 |

### 3. **Filter Data（数据过滤）**

- `filter_data=True`：推荐用于真实市场数据
- `filter_data=False`：仅在数据已预处理或质量极高时使用

---

## 📊 效果对比

### 不同正则化水平的效果

```python
# 测试不同正则化级别
for reg in [0.0, 0.01, 0.05]:
    result = smooth_calibrate_vol_surface(
        df,
        model='SABR',
        regularization=reg
    )

    surface = result['surface']
    surface.plot()  # 观察曲面平滑度
```

**预期结果：**

| Regularization | 曲面特征 | 与市场拟合 |
|---|---|---|
| 0.0 | 可能有颠簸，参数跳跃 | 拟合最紧 |
| **0.01** | **平滑，参数连续** | **平衡** ✓ |
| 0.05 | 非常平滑 | 拟合较松 |

---

## 🔍 关键改进点说明

### 1. 数据过滤改进

**问题：** 低流动性期权价格不准确

**解决：**
```python
# 移除时间价值不足的期权
df_filtered = df[df['time_value_pct'] >= 0.05]

# 限制 moneyness 范围（0.7-1.5）
df_filtered = df[(df['moneyness'] >= 0.7) & (df['moneyness'] <= 1.5)]

# Z-score 异常值检测
z_scores = zscore(df['option_price'])
df_filtered = df[z_scores < 2.5]
```

### 2. 正则化惩罚

**问题：** 相邻到期日参数跳跃过大

**解决：**
```python
# 在目标函数中添加平滑惩罚
def objective(params):
    alpha, rho, nu = params

    # 拟合误差
    fitting_error = sum((market_vol - model_vol)**2) / n

    # 平滑惩罚（惩罚参数跳跃）
    if has_previous_maturity:
        alpha_diff = (alpha - prev_alpha)**2
        rho_diff = (rho - prev_rho)**2
        nu_diff = (nu - prev_nu)**2

        smoothness_penalty = regularization * (alpha_diff + rho_diff + nu_diff)

    return fitting_error + smoothness_penalty
```

### 3. RBF 平滑插值

**问题：** `smoothing=0.0` 导致精确插值（过拟合）

**解决：**
```python
# vol_surface_calibrator.py 第 72-78 行
RBFInterpolator(
    points,
    values,
    kernel='thin_plate_spline',
    smoothing=0.005  # 从 0.0 改为 0.005
)
```

---

## ⚠️ 常见问题

### Q1: 校准后曲面仍有轻微颠簸？

**A:** 增加正则化强度
```python
result = smooth_calibrate_vol_surface(df, regularization=0.05)
```

### Q2: 曲面过于平滑，无法捕捉 smile 形状？

**A:** 降低正则化强度
```python
result = smooth_calibrate_vol_surface(df, regularization=0.005)
```

### Q3: 数据点太少，校准不稳定？

**A:**
- 放宽异常值阈值：`outlier_threshold=3.0`
- 关闭部分过滤：`filter_data=False`
- 增加子采样数量：`output_strikes=30`

### Q4: 与 Bloomberg 曲面差异大？

**A:** Bloomberg 使用的方法：
- 可能使用专有平滑技术
- 可能使用不同的 beta 参数
- 可能包含交易员调整

**建议：**
1. 先确保数据质量（使用相同数据源）
2. 调整 beta 参数（默认 0.5，可尝试 0.7）
3. 参考 Bloomberg 的 ATM vol 调整正则化强度

---

## 🎓 推荐工作流程

### Step 1: 数据清洗
```python
from option_data_cleaner import extract_options_data
df = extract_options_data("spx_data.xlsx")
```

### Step 2: 子采样（可选）
```python
from data_subsampler import subsample_options_data
df_sub = subsample_options_data(df, num_strikes=50)
```

### Step 3: 平滑校准
```python
from smooth_calibrator import smooth_calibrate_vol_surface

result = smooth_calibrate_vol_surface(
    df_sub,
    model='SABR',
    regularization=0.01,      # 根据结果调整
    filter_data=True,
    outlier_threshold=2.5
)
```

### Step 4: 检查结果
```python
surface = result['surface']

# 可视化
surface.plot()

# 检查 ATM volatility 的期限结构
import numpy as np
spot = result['spot_price']
maturities = np.linspace(0.1, 2.0, 20)
atm_vols = [surface.get_vol(spot, T) for T in maturities]

import matplotlib.pyplot as plt
plt.plot(maturities, atm_vols, 'o-')
plt.xlabel('Maturity (years)')
plt.ylabel('ATM Implied Vol')
plt.title('Term Structure of ATM Volatility')
plt.grid(True)
plt.show()
```

### Step 5: 导出到 QuantLib
```python
if QUANTLIB_AVAILABLE:
    ql_surface = surface.to_quantlib()
    # 在定价模型中使用...
```

---

## 📈 性能优化建议

1. **子采样先行**
   - 原始数据 > 2000 个期权时，先子采样到 500-1000
   - 提速 2-5 倍，平滑效果更好

2. **并行校准**
   - SABR 按 maturity slice 校准，可并行化
   - 使用 `joblib` 或 `multiprocessing`

3. **缓存结果**
   - 同一数据集的不同正则化参数可共用 implied vol 计算
   - 保存中间结果避免重复计算

---

## 📚 理论背景

### Tikhonov 正则化

目标函数：
```
minimize: ||f(x) - y||² + λ||Γx||²
         ↑                ↑
    拟合误差         正则化项
```

其中：
- `λ` = `regularization` 参数
- `Γx` = 参数变化量（参数跳跃）

**效果：**
- `λ = 0`：标准最小二乘，可能过拟合
- `λ > 0`：惩罚参数不连续性，增强平滑度
- `λ → ∞`：所有参数趋于相同（过度平滑）

### RBF 平滑参数

Radial Basis Function interpolation 中的 `smoothing`:

```
minimize: Σ||f(xi) - yi||² + smoothing * ||f||²
          ↑                   ↑
     插值误差              函数平滑度
```

- `smoothing = 0.0`：精确通过所有点（可能过拟合）
- `smoothing > 0`：允许偏离数据点以获得平滑曲面

---

## ✅ 总结

**快速决策树：**

```
数据质量好？
├─ 是 → 使用原校准器（smoothing=0.005）
└─ 否 → 使用 smooth_calibrator.py
         ├─ 正常噪声 → regularization=0.01
         ├─ 噪声大 → regularization=0.05
         └─ 数据少 → 放宽 outlier_threshold
```

**最佳实践：**
1. ✓ 总是使用数据过滤（`filter_data=True`）
2. ✓ 从 `regularization=0.01` 开始
3. ✓ 观察曲面，迭代调整
4. ✓ 与市场 ATM vol 对比验证
5. ✓ 检查参数是否跨 maturity 连续

**预期效果：**
- 平滑的波动率曲面，无异常颠簸
- 参数跨到期日变化连续
- 保留市场 smile 特征（非过度平滑）
- 与高质量数据源（如 Bloomberg）可比

---

## 📞 故障排查

### 校准失败？
- 检查数据量是否足够（每个 maturity ≥ 3 个期权）
- 尝试降低 `outlier_threshold`
- 关闭 `filter_data`

### 数值错误？
- 检查是否有负价格/strike
- 检查到期日是否在交易日之后
- 尝试更保守的参数边界

### 曲面不自然？
- 增加 `regularization`
- 检查是否有异常数据点未被过滤
- 尝试不同的 `output_strikes` 数量

---

**创建日期：** 2026-01-15
**版本：** 1.0
**适用范围：** SPX/SPY 期权波动率曲面校准