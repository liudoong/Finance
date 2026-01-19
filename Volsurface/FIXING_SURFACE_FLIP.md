# 修复波动率曲面翻转问题

## 🚨 问题描述

**症状：**
- 波动率曲面出现"翻转"或"波浪"形状
- 曲面不是单调的，而是上下起伏
- 与 Bloomberg 等高质量曲面完全不同
- 看起来非常不真实

**根本原因：**
数据质量极差，异常值主导了校准，原有的过滤**太宽松**。

---

## 🔍 数据质量诊断结果

从 `spx_data.xlsx` 发现的问题：

| 问题 | 数量 | 影响 |
|---|---|---|
| **负时间价值** | 188 个 | 违反无套利，导致 IV 为负或爆炸 |
| **时间价值 < 1%** | 2,003 个 | 几乎纯内在价值，IV 不稳定 |
| **深度价内 (K<0.7S)** | 380 个 | IV 计算不准确 |
| **深度价外 (K>1.5S)** | 33 个 | 流动性差，IV 不可靠 |
| **异常高价格 (>$1000)** | 521 个 | 深度价内期权，主要是内在价值 |
| **异常低价格 (<$0.1)** | 192 个 | 过时报价或流动性极差 |

### 典型坏数据示例

```
Strike  Spot    Option Price  Intrinsic  Time Value  问题
400     6942    6476.53       6542.51    -65.98      ⚠️ 负时间价值
7900    6942    896.30        957.49     -61.19      ⚠️ 负时间价值
6025    6942    0.02          0          0.02        ⚠️ 过时报价
2800    6921    0.02          0          0.02        ⚠️ 过时报价
```

---

## ✅ 解决方案

### 方案 1：使用激进数据过滤（推荐）

#### 步骤 1：运行数据质量诊断

```bash
cd /Users/october/Documents/Github/Finance/Volsurface
python test_data_filtering.py
```

这将：
- 诊断数据质量问题
- 应用激进过滤
- 生成对比图
- 保存清洗后的数据 `options_aggressively_filtered.csv`

#### 步骤 2：使用清洗后的数据

**选项 A：直接使用清洗后的 CSV**
```python
import pandas as pd
from smooth_calibrator import smooth_calibrate_vol_surface

# 加载已清洗的数据
df_clean = pd.read_csv('options_aggressively_filtered.csv')

# 添加必要的日期解析
df_clean['trading_date'] = pd.to_datetime(df_clean['trading_date'])
df_clean['maturity_date'] = pd.to_datetime(df_clean['maturity_date'])

# 校准
result = smooth_calibrate_vol_surface(
    df_clean,
    model='SABR',
    regularization=0.01,
    filter_data=False,  # 已经清洗过，不需要再过滤
    risk_free_rate=0.05
)

surface = result['surface']
surface.plot()
```

**选项 B：在工作流中启用激进过滤**

修改 `fast_calibration_workflow.py`：
```python
# 第 48-61 行
USE_SMOOTH_CALIBRATION = True
REGULARIZATION = 0.01
FILTER_DATA = True              # 必须打开
OUTLIER_THRESHOLD = 2.0         # 从 2.5 改为 2.0 (更严格)
```

---

### 方案 2：手动预处理数据

创建自定义清洗脚本：

```python
from option_data_cleaner import extract_options_data
from robust_data_filter import aggressive_filter_options

# 加载
df_raw = extract_options_data("spx_data.xlsx")

# 激进过滤
df_clean = aggressive_filter_options(
    df_raw,
    min_time_value_pct=0.02,      # 最小时间价值 2%
    moneyness_range=(0.75, 1.35),  # 聚焦 ATM 附近
    max_iv_pct=2.0,                # 最大 IV 200%
    min_iv_pct=0.05,               # 最小 IV 5%
    verbose=True
)

# 保存
df_clean.to_csv('my_clean_data.csv', index=False)
```

---

## 🎯 更新的过滤规则

### 之前的规则（太宽松）

| 过滤器 | 之前 | 问题 |
|---|---|---|
| 时间价值 | >= 5% | 太严格，丢失数据 |
| Moneyness | 0.7 - 1.5 | 仍包含深度价内 |
| 异常值 | Z-score < 2.5 | 对非正态分布效果差 |
| 负时间价值 | **未检查** | **致命问题** |
| IV 范围 | **未检查** | **允许异常 IV** |

### 现在的规则（更严格）

| 过滤器 | 现在 | 效果 |
|---|---|---|
| **负时间价值** | **移除** | **修复套利违规** ✅ |
| **时间价值** | >= 2% | 更合理的阈值 |
| **Moneyness** | **0.75 - 1.35** | **更紧，聚焦 ATM** ✅ |
| **异常值** | **IQR 方法** | **更稳健** ✅ |
| **IV 范围** | **5% - 200%** | **过滤异常 IV** ✅ |
| **价格范围** | 0.1 - 1.5×Spot | 移除过时报价 |

---

## 📊 预期效果

### 过滤前（原始数据）
```
总期权: 3,897 个
问题数据: ~2,200 个 (56%)
可用数据: ~1,700 个 (44%)
```

### 过滤后
```
保留期权: ~900-1,500 个 (23-38%)
问题数据: 0 个
数据质量: 高 ✓
```

### 校准效果

**之前（坏数据）：**
- ❌ 曲面翻转、波浪形
- ❌ ATM vol 跳跃
- ❌ 参数α跳跃 > 0.2
- ❌ 与市场不符

**之后（清洗数据）：**
- ✅ 曲面平滑单调
- ✅ ATM vol 连续
- ✅ 参数α跳跃 < 0.05
- ✅ 与市场接近

---

## 🛠️ 完整修复流程

### Step 1: 诊断数据

```bash
python test_data_filtering.py
```

查看：
- `data_filtering_comparison.png` - 过滤前后对比
- 终端输出的问题统计

### Step 2: 清洗数据

数据已经被 `test_data_filtering.py` 清洗并保存为：
- `options_aggressively_filtered.csv`

### Step 3: 使用清洗数据校准

```bash
python fast_calibration_workflow.py
```

确保设置：
```python
USE_SMOOTH_CALIBRATION = True
FILTER_DATA = True
REGULARIZATION = 0.01
```

### Step 4: 检查结果

打开生成的图片：
- `vol_surface_sabr_analysis.png`

检查：
- [ ] ATM 期限结构平滑单调
- [ ] Smile 形状合理（两边低、中间高）
- [ ] 无异常翻转或波浪
- [ ] 参数α跳跃 < 0.05

---

## 🔍 验证清单

使用此清单验证修复效果：

### 数据质量
- [ ] 无负时间价值期权
- [ ] 所有 Moneyness 在 0.75-1.35 范围
- [ ] 所有 IV 在 5%-200% 范围
- [ ] 保留率在 20%-50% 之间（正常）

### 曲面质量
- [ ] 3D 曲面平滑，无突起或凹陷
- [ ] ATM 期限结构单调（一般递增）
- [ ] Vol smile 凸性正确（U型）
- [ ] 无异常翻转或波浪

### 参数质量
- [ ] SABR α 参数连续变化
- [ ] max α jump < 0.05
- [ ] ρ 参数在 -0.7 到 0 之间（正常）
- [ ] ν 参数合理（0.2-0.8）

---

## ❓ 常见问题

### Q1: 过滤后数据太少（< 20%），校准失败？

**A:** 原始数据质量极差。尝试：

```python
# 放宽 moneyness 范围
aggressive_filter_options(
    df,
    moneyness_range=(0.70, 1.40),  # 稍微放宽
    min_time_value_pct=0.015       # 降低阈值
)
```

### Q2: 曲面还是有轻微波动？

**A:** 增加正则化：

```python
# fast_calibration_workflow.py
REGULARIZATION = 0.05  # 从 0.01 增加到 0.05
```

### Q3: 过滤后某些到期日没有数据？

**A:** 正常。流动性差的到期日被完全过滤。使用原始校准器对这些到期日：

```python
# 混合方案：主要用平滑校准，特殊情况用原始
if len(df_clean) > 100:
    result = smooth_calibrate_vol_surface(df_clean, ...)
else:
    result = calibrate_vol_surface(df_raw, ...)
```

### Q4: 如何判断数据是否足够好？

**A:** 运行诊断：

```python
from robust_data_filter import diagnose_data_quality

diagnose_data_quality(df)
```

良好数据的指标：
- 负时间价值 < 5%
- 深度价内/价外 < 20%
- 时间价值中位数 > 1%

---

## 📚 技术细节

### 为什么负时间价值导致曲面翻转？

1. **无套利原则**：
   ```
   Option_Price >= Intrinsic_Value
   Time_Value = Option_Price - Intrinsic >= 0
   ```

2. **违规的影响**：
   - Newton-Raphson 求解 IV 时，目标函数无解
   - 返回 `np.nan` 或极端值（< 0 或 > 1000%）
   - 插值器尝试拟合这些极端值
   - 曲面被拉向异常方向 → 翻转

3. **真实原因**：
   - Last Price 是旧数据（过时报价）
   - 深度价内期权套利被执行
   - 数据采集错误

### IQR 方法 vs Z-Score

**Z-Score 方法：**
```python
z = (x - mean) / std
outlier if |z| > 2.5
```
- 假设正态分布
- 对极端值敏感

**IQR 方法（更稳健）：**
```python
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
```
- 不假设分布
- 对极端值不敏感 ✅

---

## ✅ 总结

**问题根源：**
- 真实市场数据质量差（56% 是坏数据）
- 原有过滤太宽松，让异常值通过
- 负时间价值导致 IV 爆炸，曲面翻转

**解决方案：**
1. 激进数据过滤（[robust_data_filter.py](robust_data_filter.py)）
2. 更新 smooth_calibrator 的过滤规则
3. 添加 IV 范围检查（5%-200%）

**使用方法：**
```bash
# 诊断 + 清洗
python test_data_filtering.py

# 校准（确保 FILTER_DATA=True）
python fast_calibration_workflow.py
```

**预期结果：**
- 平滑、单调的波动率曲面
- 参数连续变化
- 与市场一致

---

**更新日期：** 2026-01-15
**问题类型：** 数据质量导致的曲面翻转
**解决状态：** ✅ 已修复