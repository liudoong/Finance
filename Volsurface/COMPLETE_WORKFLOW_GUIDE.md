# 完整的波动率曲面校准工作流指南

## 🎯 概览

本项目提供了从原始期权数据到校准波动率曲面的完整工作流，特别针对数据质量问题进行了优化。

## 📂 核心文件结构

### 1. 数据处理模块

| 文件 | 用途 | 状态 |
|------|------|------|
| **option_data_cleaner.py** | 从 Excel 提取原始数据 | ✅ 已更新（新格式） |
| **quality_sampler.py** | 质量优先采样（采样+过滤一体化） | ⭐ 推荐使用 |
| **data_subsampler.py** | 原始采样器（仅采样） | ⚠️ 已过时 |
| **ultra_robust_filter.py** | 超级激进过滤器（事后过滤） | ⚠️ 备选方案 |

### 2. 校准模块

| 文件 | 用途 | 状态 |
|------|------|------|
| **clean_workflow.py** | 清洁工作流（质量采样→SABR校准） | ⭐ 推荐使用 |
| **vol_surface_calibrator.py** | 核心校准引擎 | ✅ 生产级别 |
| **smooth_calibrator.py** | 带正则化的校准器 | ⚠️ 复杂，不推荐 |
| **fast_calibration_workflow.py** | 快速校准流程 | ⚠️ 已过时 |

### 3. 可视化模块

| 文件 | 用途 | 状态 |
|------|------|------|
| **plot_market_iv_surface.py** | 可视化原始市场IV（无校准） | ⭐ 诊断工具 |
| **plot_vol_surface.py** | 可视化校准后的曲面 | ✅ 生产级别 |

### 4. 文档

| 文件 | 内容 |
|------|------|
| **README_CLEAN_WORKFLOW.md** | 清洁工作流详细指南 |
| **RUN_MARKET_IV_PLOT.md** | 市场IV可视化使用说明 |
| **FIXING_SURFACE_FLIP.md** | 曲面翻转问题诊断与修复 |
| **SMOOTHING_GUIDE.md** | 平滑参数调优指南 |
| **COMPLETE_WORKFLOW_GUIDE.md** | 本文档 |

---

## 🚀 推荐工作流（从零开始）

### Step 1: 诊断原始数据质量

**目的：** 了解数据源是否可靠

```bash
cd /Users/october/Documents/Github/Finance/Volsurface
python3 plot_market_iv_surface.py
```

**查看输出：**
- 打开 `market_iv_surface.png`
- 检查终端输出的统计信息

**判断标准：**

✅ **数据质量好**（可以直接校准）：
- IV 范围在 10%-60% 之间
- 期限结构大致单调递增
- 波动率微笑形状规则（U型或平坦）
- 无极端离群值

❌ **数据质量差**（需要严格过滤）：
- IV 范围包含 <5% 或 >80% 的异常值
- 期限结构有 V 型坑或极端跳跃
- 某些到期日数据极少或极多
- 3D 图显示明显的波浪/尖刺

---

### Step 2: 运行清洁工作流（推荐）

**目的：** 使用质量优先采样进行校准

```bash
python3 clean_workflow.py
```

**这会自动完成：**
1. ✅ 加载原始数据
2. ✅ 质量优先采样（同时过滤）
3. ✅ SABR 模型校准
4. ✅ 生成可视化图表
5. ✅ 保存所有结果

**预期时间：** 30-60 秒

**输出文件：**
- `options_quality_sampled.csv` - 高质量期权数据
- `vol_surface_grid_clean.csv` - 波动率曲面网格
- `sabr_params_clean.csv` - SABR 参数
- `vol_surface_clean_analysis.png` - 曲面分析图 ⭐
- `sabr_params_clean.png` - 参数演化图

---

### Step 3: 检查校准结果

**打开：** `vol_surface_clean_analysis.png`

**检查清单：**

| 子图 | 检查项 | 目标 |
|------|--------|------|
| **3D 曲面** | 是否平滑无刺 | 应该像光滑的山丘 |
| **Term Structure** | 是否单调递增 | 不应有 V 型坑 |
| **Volatility Smile** | 形状是否规则 | U型或平坦，不应有尖刺 |
| **市场 vs 模型** | 拟合是否良好 | 点应在线附近 |
| **残差** | 是否随机分布 | 不应有系统性偏差 |

**查看终端输出：**

```
质量检查:
  IV 范围: 12.5% - 58.3%  ✓
  参数跳跃: 0.0234        ✓

数据流转:
  3897 个原始期权
      ↓ (质量优先采样)
  624 个高质量期权 (16.0%)
      ↓ (SABR校准)
  80 × 15 波动率曲面
```

**成功标准：**
- ✅ 保留率 10%-30%
- ✅ IV 范围 10%-60%
- ✅ 参数跳跃 < 0.05
- ✅ 无质量警告

---

### Step 4: 如果结果不理想

#### 场景 A: 保留率太低（<10%）

**原因：** 参数过于严格

**解决：** 编辑 `clean_workflow.py` 第 27-33 行

```python
# 放宽参数
MONEYNESS_RANGE = (0.80, 1.25)  # 从 (0.85, 1.20) 扩大
MIN_IV = 0.08                   # 从 0.10 降低
MAX_IV = 0.70                   # 从 0.60 提高
MIN_TIME_VALUE_PCT = 0.02       # 从 0.03 降低
```

重新运行：`python3 clean_workflow.py`

---

#### 场景 B: 曲面仍有轻微波动

**原因：** IV 范围太宽，包含异常值

**解决：** 收紧 IV 范围

```python
# 更严格的 IV 过滤
MIN_IV = 0.12  # 从 0.10 提高
MAX_IV = 0.50  # 从 0.60 降低
```

重新运行：`python3 clean_workflow.py`

---

#### 场景 C: 某些到期日被完全放弃

**原因：** 这些到期日数据质量极差（这是正常的！）

**说明：** 质量差的到期日会破坏整个曲面，放弃它们是正确的

**如果想保留更多：**

```python
MIN_OPTIONS_PER_MATURITY = 10  # 从 15 降低
```

但要注意可能引入噪声。

---

#### 场景 D: 期限结构有极端跳跃

**可能原因：**
1. 原始数据本身就有问题（检查 Step 1 的市场 IV 图）
2. 某些到期日数据质量极差但未被过滤

**解决方案 1：** 使用超级激进过滤器

```bash
python3 ultra_robust_filter.py
```

这会生成 `options_ultra_clean.csv`，然后用这个文件手动校准。

**解决方案 2：** 手动移除问题到期日

打开 `options_quality_sampled.csv`，检查哪些到期日的 IV 异常，手动删除后重新校准。

---

## 🎛️ 参数调优决策树

```
开始：运行 clean_workflow.py
  ↓
结果评估
  │
  ├─ ✅ 曲面平滑，期限结构单调
  │   → 完成！使用这些参数
  │
  ├─ ❌ 保留率 < 10%（数据太少）
  │   → 放宽 MONEYNESS_RANGE, MIN_IV, MAX_IV
  │   → 重新运行
  │
  ├─ ❌ 曲面有轻微波动
  │   → 收紧 MIN_IV/MAX_IV (例如 0.12-0.50)
  │   → 重新运行
  │
  ├─ ❌ 期限结构有跳跃
  │   → 检查 plot_market_iv_surface.py 输出
  │   │
  │   ├─ 原始市场 IV 就有跳跃
  │   │   → 数据源问题，使用 ultra_robust_filter.py
  │   │
  │   └─ 原始市场 IV 平滑
  │       → 调整 MIN_OPTIONS_PER_MATURITY = 20
  │       → 移除质量差的到期日
  │
  └─ ❌ 曲面翻转/波浪（严重问题）
      → 参考 FIXING_SURFACE_FLIP.md
      → 使用 ultra_robust_filter.py + clean_workflow.py
```

---

## 🔧 高级用法

### 自定义质量采样

如果您需要完全控制采样过程：

```python
from quality_sampler import quality_subsample_options
from option_data_cleaner import extract_options_data

# 加载数据
df_raw = extract_options_data("spx_data.xlsx")

# 自定义采样
df_quality = quality_subsample_options(
    df_raw,
    num_strikes_per_maturity=50,      # 更多数据点
    moneyness_range=(0.90, 1.15),     # 更聚焦 ATM
    min_iv=0.15,                      # 更严格的下限
    max_iv=0.45,                      # 更严格的上限
    min_time_value_pct=0.04,          # 更高的时间价值要求
    min_options_per_maturity=20,      # 更高的质量门槛
    risk_free_rate=0.05
)

# 保存并手动校准
df_quality.to_csv('my_custom_sample.csv', index=False)
```

然后使用 `vol_surface_calibrator.py` 手动校准这个数据。

---

### 对比不同配置

创建一个测试脚本：

```python
import pandas as pd
from quality_sampler import quality_subsample_options
from option_data_cleaner import extract_options_data
from vol_surface_calibrator import calibrate_vol_surface

df_raw = extract_options_data("spx_data.xlsx")

configs = [
    {"name": "宽松", "moneyness": (0.75, 1.30), "iv_range": (0.08, 0.80)},
    {"name": "标准", "moneyness": (0.85, 1.20), "iv_range": (0.10, 0.60)},
    {"name": "严格", "moneyness": (0.90, 1.15), "iv_range": (0.12, 0.50)},
]

for config in configs:
    print(f"\n测试配置: {config['name']}")

    df_quality = quality_subsample_options(
        df_raw,
        moneyness_range=config['moneyness'],
        min_iv=config['iv_range'][0],
        max_iv=config['iv_range'][1]
    )

    result = calibrate_vol_surface(df_quality, model='SABR')

    # 评估结果...
```

---

## 📊 输出文件说明

### options_quality_sampled.csv

**内容：** 经过质量采样的期权数据

**列：**
- `option_contract`: 期权合约名
- `trading_date`: 交易日期
- `maturity_date`: 到期日
- `strike`: 行权价
- `option_type`: 期权类型（Call/Put）
- `option_price`: 期权价格
- `spot_price`: 现货价格
- `implied_vol`: 隐含波动率（已计算）
- `time_to_maturity`: 到期时间（年）

**用途：** 可以用于其他分析或手动校准

---

### vol_surface_grid_clean.csv

**内容：** 校准后的波动率曲面网格

**列：**
- `Strike`: 行权价
- `Maturity_Years`: 到期时间（年）
- `Implied_Vol`: 隐含波动率
- `Moneyness`: K/S 比率

**用途：**
- 导入其他系统使用
- 与市场数据对比
- 风险管理计算

---

### sabr_params_clean.csv

**内容：** 每个到期日的 SABR 参数

**列：**
- `Maturity_Years`: 到期时间
- `alpha`: 初始波动率水平
- `beta`: CEV 参数（通常固定为 0.5）
- `rho`: 波动率与标的的相关性
- `nu`: 波动率的波动率（vol-of-vol）

**用途：**
- 理解模型行为
- 检查参数连续性
- 用于定价新期权

---

## 🐛 常见问题

### Q1: "没有找到任何高质量数据"

**原因：** 参数太严格或数据质量极差

**解决：**
1. 检查原始数据（运行 `plot_market_iv_surface.py`）
2. 放宽参数（参考"场景 A"）
3. 如果还不行，检查数据源是否正确

---

### Q2: 运行时出现 "ModuleNotFoundError"

**原因：** 缺少 Python 库

**解决：**
```bash
pip install pandas numpy scipy matplotlib openpyxl
```

或使用 conda：
```bash
conda install pandas numpy scipy matplotlib openpyxl
```

---

### Q3: 校准后 Min IV 仍然是 1%

**原因：** 采样阶段的过滤没有生效

**诊断：**
1. 检查 `options_quality_sampled.csv` 中的 `implied_vol` 列
2. 如果该列最小值确实是 0.01，说明采样器有问题
3. 如果该列正常（>0.10），但校准后异常，说明校准器有问题

**解决：**
- 场景 1: 手动编辑 `quality_sampler.py`，增加 `min_iv` 参数
- 场景 2: 使用 `ultra_robust_filter.py` 预处理

---

### Q4: 曲面仍然翻转

**可能原因：**
1. 原始数据有大量负时间价值期权
2. 某些到期日数据完全错误
3. SABR 校准陷入局部最优

**解决方案：**
1. 运行 `ultra_robust_filter.py` 获得 `options_ultra_clean.csv`
2. 手动检查该文件，确认 IV 范围合理
3. 如果还不行，考虑更换数据源或只使用流动性最好的期权（ATM, 1-3个月到期）

---

## 💡 最佳实践

### ✅ 推荐做法

1. **始终先可视化原始数据**
   ```bash
   python3 plot_market_iv_surface.py
   ```
   理解数据质量再决定策略

2. **使用清洁工作流作为基线**
   ```bash
   python3 clean_workflow.py
   ```
   默认参数适合大多数情况

3. **逐步调整，不要一次改太多**
   - 每次只改一个参数
   - 观察效果
   - 记录最佳配置

4. **保存中间结果**
   - `options_quality_sampled.csv` 可以重复使用
   - 避免每次都重新采样

5. **对比市场 vs 模型**
   - 校准后的曲面应该比原始数据平滑
   - 但不应过度平滑（丢失市场信息）

---

### ❌ 避免做法

1. **不要跳过数据诊断步骤**
   - 盲目校准会浪费时间
   - 先了解数据质量

2. **不要使用过时的脚本**
   - ~~fast_calibration_workflow.py~~（事后过滤）
   - ~~data_subsampler.py~~（只采样不过滤）
   - 使用 `clean_workflow.py`

3. **不要忽视质量警告**
   - 终端输出的警告很重要
   - 参数跳跃 > 0.10 说明有严重问题

4. **不要期望完美曲面**
   - 真实市场数据有噪声
   - 轻微的波动是正常的
   - 目标是"合理平滑"，不是"完全平坦"

---

## 📚 相关文档索引

| 主题 | 文档 |
|------|------|
| 质量优先采样详细说明 | [README_CLEAN_WORKFLOW.md](README_CLEAN_WORKFLOW.md) |
| 市场 IV 可视化使用 | [RUN_MARKET_IV_PLOT.md](RUN_MARKET_IV_PLOT.md) |
| 曲面翻转问题修复 | [FIXING_SURFACE_FLIP.md](FIXING_SURFACE_FLIP.md) |
| 平滑参数调优 | [SMOOTHING_GUIDE.md](SMOOTHING_GUIDE.md) |

---

## 🎓 总结

### 核心理念

**"垃圾进，垃圾出"** - 数据质量决定一切

**解决方案：**
1. ✅ 在采样阶段就确保质量（`quality_sampler.py`）
2. ✅ 提前计算和验证 IV
3. ✅ 移除整个质量差的到期日
4. ✅ 使用质量评分智能选择期权

### 工作流总结

```
原始数据 (spx_data.xlsx)
    ↓
[可选] 可视化诊断 (plot_market_iv_surface.py)
    ↓
质量优先采样 (quality_sampler.py)
    ↓
SABR 校准 (vol_surface_calibrator.py)
    ↓
可视化分析 (plot_vol_surface.py)
    ↓
结果评估 & 参数调优
    ↓
生产使用
```

### 立即开始

```bash
cd /Users/october/Documents/Github/Finance/Volsurface

# 步骤 1: 诊断原始数据
python3 plot_market_iv_surface.py

# 步骤 2: 清洁工作流校准
python3 clean_workflow.py

# 步骤 3: 检查结果
open vol_surface_clean_analysis.png
open market_iv_surface.png
```

---

**版本：** 3.0 (Complete Workflow)
**更新日期：** 2026-01-15
**状态：** ✅ 生产就绪
**维护者：** Claude Code
**许可：** Internal Use

---

## 🆘 需要帮助？

如果遇到问题：

1. **检查文档：** 先查看相关 .md 文件
2. **查看输出：** 终端输出通常包含有用的提示
3. **对比结果：** 市场 IV vs 校准曲面
4. **逐步调试：** 从简单参数开始，逐步收紧

**记住：** 平滑的波动率曲面来自高质量的数据，而不是复杂的算法。

---

**祝您校准顺利！** 🚀