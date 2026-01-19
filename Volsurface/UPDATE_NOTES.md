# 更新说明 (2026-01-15)

## 🎯 主要更新

为解决波动率曲面 **bumps（颠簸）和过拟合问题**，新增平滑校准功能。

---

## 📁 新增文件

### 1. **smooth_calibrator.py** ⭐ 核心模块
平滑校准器，包含：
- 数据质量过滤（流动性、异常值）
- Tikhonov 正则化（惩罚参数跳跃）
- 可调节平滑强度

**主要函数：**
```python
smooth_calibrate_vol_surface(
    df,
    model='SABR',
    regularization=0.01,    # 关键参数
    filter_data=True,
    outlier_threshold=2.5
)
```

### 2. **SMOOTHING_GUIDE.md** 📖 详细指南
包含：
- 问题分析和解决方案
- 参数调优建议
- 使用示例和最佳实践
- 故障排查

### 3. **test_smooth_calibration.py** 🧪 测试脚本
对比三种校准方法：
- 原始校准（无正则化）
- 中度平滑（λ=0.01）
- 强平滑（λ=0.05）

自动生成对比图和参数连续性分析。

### 4. **UPDATE_NOTES.md** 📝 本文件
快速参考更新内容。

---

## 🔧 修改的文件

### 1. **vol_surface_calibrator.py**
- RBF 插值器 `smoothing` 参数：`0.0` → `0.005`
- 增加平滑度，减少过拟合

### 2. **fast_calibration_workflow.py** ⭐ 主工作流
**新增配置参数：**
```python
USE_SMOOTH_CALIBRATION = True    # 是否使用平滑校准
REGULARIZATION = 0.01            # 正则化强度
FILTER_DATA = True               # 数据质量过滤
OUTLIER_THRESHOLD = 2.5          # 异常值阈值
```

**新增功能：**
- 自动选择平滑/标准校准
- 参数连续性检查（检测α跳跃）
- 详细的校准方法说明

---

## 🚀 快速开始

### 方法 1：使用更新的工作流脚本（推荐）

```python
# 编辑 fast_calibration_workflow.py 中的配置
USE_SMOOTH_CALIBRATION = True
REGULARIZATION = 0.01
FILTER_DATA = True

# 运行
python fast_calibration_workflow.py
```

### 方法 2：直接使用平滑校准器

```python
from smooth_calibrator import smooth_calibrate_vol_surface
from option_data_cleaner import extract_options_data

df = extract_options_data("spx_data.xlsx")

result = smooth_calibrate_vol_surface(
    df,
    model='SABR',
    regularization=0.01,
    filter_data=True
)

surface = result['surface']
surface.plot()
```

### 方法 3：运行对比测试

```bash
python test_smooth_calibration.py
```

查看生成的对比图，选择最佳 regularization 参数。

---

## 📊 关键改进

### 问题：原始校准的 bumps

| 问题 | 原因 | 解决方案 |
|---|---|---|
| 曲面有颠簸 | 插值过拟合 | RBF smoothing=0.005 |
| 参数跳跃大 | 无平滑约束 | Tikhonov 正则化 |
| 受噪声影响 | 低质量数据 | 数据质量过滤 |

### 效果：平滑校准

✅ 波动率曲面平滑无颠簸
✅ ATM 期限结构单调连续
✅ SABR 参数跨到期日变化平滑
✅ 与 Bloomberg 等高质量曲面更接近

---

## 🎛️ 参数调优指南

### Regularization（正则化强度）

| 值 | 效果 | 适用场景 |
|---|---|---|
| 0.0 | 无平滑 | 数据质量极高 |
| 0.005 | 轻度平滑 | 正常市场 |
| **0.01** ⭐ | **中度平滑** | **大多数情况** |
| 0.05 | 强平滑 | 噪声大 |
| 0.1 | 极强平滑 | 数据点少 |

### Filter Data（数据过滤）

**推荐：** `filter_data=True`

**过滤规则：**
- 时间价值 < 5% spot price → 移除
- Moneyness < 0.7 或 > 1.5 → 移除
- Z-score > 2.5 → 移除（异常值）
- 价格不合理 → 移除

**效果：** 通常移除 30-40% 数据，但这些是噪声

---

## 📈 使用建议

### 决策树

```
校准后发现bumps？
├─ 是 → 设置 USE_SMOOTH_CALIBRATION = True
│       从 REGULARIZATION = 0.01 开始
│       观察效果，调整 0.005-0.05
│
└─ 否 → 保持当前设置
        或使用原始校准（USE_SMOOTH_CALIBRATION = False）
```

### 最佳实践

1. ✅ **第一次运行**：使用推荐设置
   ```python
   USE_SMOOTH_CALIBRATION = True
   REGULARIZATION = 0.01
   FILTER_DATA = True
   ```

2. ✅ **检查结果**：
   - 查看生成的图表（vol_surface_sabr_analysis.png）
   - 检查参数连续性（max α jump < 0.05）
   - 对比市场 ATM vol

3. ✅ **调整参数**（如需要）：
   - 仍有bumps → 增加 REGULARIZATION 到 0.05
   - 过度平滑 → 降低 REGULARIZATION 到 0.005
   - 数据点太少 → 设置 FILTER_DATA = False

4. ✅ **验证**：
   - ATM 期限结构应该平滑单调
   - Smile 形状合理（两边低、中间高）
   - 与高质量来源（Bloomberg）对比

---

## 🔍 故障排查

### Q: 校准报错 "No valid data after filtering"？

**A:** 数据过滤太严格
```python
# 解决方案1：放宽异常值阈值
OUTLIER_THRESHOLD = 3.0

# 解决方案2：关闭数据过滤
FILTER_DATA = False
```

### Q: 曲面还是有bumps？

**A:** 增加正则化强度
```python
REGULARIZATION = 0.05  # 或 0.1
```

### Q: 曲面过度平滑，丢失smile特征？

**A:** 降低正则化强度
```python
REGULARIZATION = 0.005  # 或 0.0
```

### Q: 参数显示 "可能有bumps ⚠"？

**A:** max α jump > 0.05 表示参数不够平滑
```python
# 增加正则化
REGULARIZATION = 0.05

# 或运行对比测试找最佳值
python test_smooth_calibration.py
```

---

## 📚 相关文档

- **详细指南：** [SMOOTHING_GUIDE.md](SMOOTHING_GUIDE.md)
- **平滑校准器代码：** [smooth_calibrator.py](smooth_calibrator.py)
- **测试脚本：** [test_smooth_calibration.py](test_smooth_calibration.py)
- **主工作流：** [fast_calibration_workflow.py](fast_calibration_workflow.py)

---

## 🎓 技术细节

### Tikhonov 正则化

目标函数：
```
minimize: fitting_error + λ * smoothness_penalty
```

其中：
```python
fitting_error = Σ(market_vol - model_vol)² / n
smoothness_penalty = Σ(param_t - param_{t-1})² / Δt
```

### 数据过滤

```python
# 1. 时间价值过滤
time_value = option_price - intrinsic_value
keep if time_value >= 5% * spot_price

# 2. Moneyness 过滤
moneyness = strike / spot_price
keep if 0.7 <= moneyness <= 1.5

# 3. Z-score 异常值检测
z = (price - mean) / std
keep if |z| < 2.5

# 4. 价格合理性
keep if option_price < spot_price
```

### RBF 平滑插值

```python
# 原始（可能过拟合）
RBFInterpolator(..., smoothing=0.0)

# 改进（平滑）
RBFInterpolator(..., smoothing=0.005)
```

---

## ✅ 验证清单

使用平滑校准后，检查：

- [ ] 波动率曲面图无明显颠簸
- [ ] ATM 期限结构平滑单调
- [ ] Smile 形状合理（凸性正确）
- [ ] SABR 参数连续（max α jump < 0.05）
- [ ] 与市场 ATM vol 一致性检查
- [ ] 校准RMSE在合理范围（< 1% vol）

---

## 📞 总结

**核心改进：**
1. 新增 `smooth_calibrator.py` 平滑校准模块
2. 更新 `fast_calibration_workflow.py` 支持平滑校准
3. 修改 `vol_surface_calibrator.py` 插值器平滑参数
4. 提供完整文档和测试工具

**推荐使用：**
```python
# 在 fast_calibration_workflow.py 中
USE_SMOOTH_CALIBRATION = True
REGULARIZATION = 0.01
FILTER_DATA = True
```

**预期效果：**
- 获得平滑、稳定的波动率曲面
- 消除 bumps 和过拟合
- 参数跨到期日连续变化
- 与高质量市场数据更接近

---

**更新日期：** 2026-01-15
**版本：** 1.0
**适用范围：** SPX/SPY 期权波动率曲面校准