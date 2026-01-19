```markdown
# 清洁工作流 - 从根本解决曲面质量问题

## 🎯 核心理念

**问题根源：** 之前的方法是"先采样，后过滤"，导致坏数据进入校准。

**新的解决方案：** **在采样阶段就彻底清洗数据**，确保只有高质量期权进入校准。

---

## 📁 新的文件结构

### 核心模块

1. **quality_sampler.py** ⭐ 质量优先采样器
   - 采样 + 过滤一体化
   - 计算 IV 并验证
   - 质量评分和智能选择
   - 移除整个质量差的到期日

2. **clean_workflow.py** ⭐ 清洁工作流
   - 简化的端到端流程
   - 使用质量采样器
   - 直接用原始校准器（数据已干净）
   - 详细的质量报告

### 旧文件（不再推荐）

- ~~fast_calibration_workflow.py~~ - 复杂，事后过滤效果差
- ~~smooth_calibrator.py~~ - 过滤逻辑分散
- ~~data_subsampler.py~~ - 只采样不过滤

---

## 🚀 使用方法

### 方法 1：直接运行清洁工作流（推荐）

```bash
cd /Users/october/Documents/Github/Finance/Volsurface
python3 clean_workflow.py
```

**这个脚本会：**
1. 加载原始数据
2. 质量采样（同时过滤）
3. 校准 SABR 模型
4. 生成曲面图
5. 保存所有结果

**预期输出：**
```
原始数据: 3897 个期权
  ↓ 质量采样
高质量数据: 400-800 个期权 (10-20%)
  ↓ SABR 校准
平滑的波动率曲面 ✓
```

### 方法 2：自定义参数

编辑 `clean_workflow.py` 第 19-29 行：

```python
# 严格模式（最平滑，但数据少）
NUM_STRIKES_PER_MATURITY = 30
MONEYNESS_RANGE = (0.90, 1.15)
MIN_IV = 0.12
MAX_IV = 0.50

# 宽松模式（数据多，但可能有轻微波动）
NUM_STRIKES_PER_MATURITY = 50
MONEYNESS_RANGE = (0.80, 1.25)
MIN_IV = 0.08
MAX_IV = 0.70
```

### 方法 3：单独使用质量采样器

```python
from quality_sampler import quality_subsample_options
from option_data_cleaner import extract_options_data

# 加载
df_raw = extract_options_data("spx_data.xlsx")

# 质量采样
df_quality = quality_subsample_options(
    df_raw,
    num_strikes_per_maturity=40,
    moneyness_range=(0.85, 1.20),
    min_iv=0.10,
    max_iv=0.60
)

# 保存
df_quality.to_csv('my_quality_data.csv', index=False)
```

---

## 🎛️ 参数说明

### 质量采样参数

| 参数 | 默认值 | 说明 | 调优建议 |
|---|---|---|---|
| `num_strikes_per_maturity` | 40 | 每个到期日目标期权数 | 更多 = 更详细，但可能引入噪声 |
| `min_time_value_pct` | 0.03 (3%) | 最小时间价值 | 提高 = 更严格，移除深度价内 |
| `moneyness_range` | (0.85, 1.20) | Moneyness 范围 | 收窄 = 更聚焦 ATM，更平滑 |
| `min_iv` | 0.10 (10%) | 最小 IV | **关键：** 避免异常低 IV |
| `max_iv` | 0.60 (60%) | 最大 IV | **关键：** 避免异常高 IV |
| `min_options_per_maturity` | 15 | 每到期日最少期权 | 太少的到期日直接放弃 |

### 调优决策树

```
曲面有波动？
├─ Term Structure 跳跃 → 收紧 MIN_IV/MAX_IV
│  例如：MIN_IV=0.12, MAX_IV=0.50
│
├─ Smile 不规则 → 收窄 MONEYNESS_RANGE
│  例如：(0.90, 1.15)
│
└─ 数据太少 → 放宽参数
   例如：MONEYNESS_RANGE=(0.80, 1.25)
        MIN_IV=0.08
```

---

## 📊 质量指标

### 好的结果应该满足：

| 指标 | 目标范围 | 说明 |
|---|---|---|
| **保留率** | 10-30% | 太高说明过滤不够，太低说明数据太差 |
| **IV 范围** | 10%-60% | 超出说明有异常值 |
| **IV 标准差** | < 10% | 太大说明数据不一致 |
| **到期日保留率** | > 70% | 太低说明很多到期日被放弃 |
| **参数α跳跃** | < 0.05 | 确保曲面平滑 |
| **Min Vol** | > 8% | 确保没有异常低波动率 |

### 运行后检查清单

- [ ] Terminal 输出的"质量检查"没有警告
- [ ] IV 范围在 10%-60% 之间
- [ ] 参数α跳跃 < 0.05
- [ ] Term Structure 平滑单调（图中无V型坑）
- [ ] Smile 形状规则（U型或平坦）
- [ ] 3D 曲面平滑无刺

---

## 🔍 故障排查

### Q1: "没有找到任何高质量数据"

**原因：** 参数太严格

**解决：**
```python
# 放宽参数
MONEYNESS_RANGE = (0.75, 1.30)
MIN_IV = 0.08
MAX_IV = 0.80
MIN_OPTIONS_PER_MATURITY = 10
```

### Q2: 曲面还有轻微波动

**原因：** IV 范围太宽

**解决：**
```python
# 收紧 IV 范围
MIN_IV = 0.12  # 从 0.10 提高
MAX_IV = 0.50  # 从 0.60 降低
```

### Q3: 某些到期日被完全放弃

**原因：** 这些到期日数据质量太差（正常）

**解决：** 这是好事！质量差的到期日会破坏整个曲面。如果想保留更多：
```python
MIN_OPTIONS_PER_MATURITY = 10  # 从 15 降低
```

### Q4: 保留率太低（<10%）

**原因：** 原始数据质量极差

**解决：**
1. 检查数据源（是否是正确的文件）
2. 放宽 `min_time_value_pct` 到 0.02
3. 考虑使用其他数据源

---

## 📈 预期效果

### 之前（fast_calibration_workflow）

```
原始数据: 3897 个
  ↓ 子采样
1200 个（包含坏数据）
  ↓ 事后过滤
800 个（但已经太晚）
  ↓ 校准
曲面有波动 ✗
```

**问题：** 坏数据已经影响了子采样，事后过滤来不及

### 现在（clean_workflow）

```
原始数据: 3897 个
  ↓ 质量采样（同时过滤）
600 个高质量期权 ✓
  ↓ 校准
平滑曲面 ✓
```

**优势：** 采样阶段就确保质量，校准用的都是好数据

---

## 💡 技术细节

### 质量评分算法

```python
quality_score =
    3.0 / (1.0 + atm_distance) +  # ATM 优先（权重最高）
    iv_score +                     # IV 合理性
    tv_score                       # 时间价值充足性

# 然后在 moneyness 桶内选择质量最高的期权
```

### 为什么这样工作

1. **ATM 优先：** ATM 期权流动性最好、定价最准
2. **IV 验证：** 提前计算 IV，移除计算失败的
3. **按桶采样：** 确保 moneyness 均匀分布
4. **到期日质量检查：** 整个到期日质量差就放弃

---

## ✅ 使用建议

### 首次使用

1. 直接运行 `clean_workflow.py`
2. 查看 `vol_surface_clean_analysis.png`
3. 检查 Term Structure 是否平滑

### 如果效果不好

1. 查看 Terminal 输出的"质量警告"
2. 根据提示调整参数
3. 重新运行

### 生产使用

```python
# 推荐的稳健配置
NUM_STRIKES_PER_MATURITY = 40
MONEYNESS_RANGE = (0.85, 1.20)
MIN_IV = 0.10
MAX_IV = 0.60
MIN_TIME_VALUE_PCT = 0.03
```

---

## 📚 相关文档

- **质量采样器代码：** [quality_sampler.py](quality_sampler.py)
- **清洁工作流：** [clean_workflow.py](clean_workflow.py)
- **修复曲面翻转指南：** [FIXING_SURFACE_FLIP.md](FIXING_SURFACE_FLIP.md)

---

## 🎓 总结

**核心改进：**
1. ✅ 采样和过滤同时进行
2. ✅ 质量评分智能选择
3. ✅ 提前计算和验证 IV
4. ✅ 移除质量差的整个到期日

**预期结果：**
- 平滑单调的 Term Structure
- 规则的 Volatility Smile
- 连续的 SABR 参数
- 无异常波动的曲面

**立即开始：**
```bash
python3 clean_workflow.py
```

---

**更新日期：** 2026-01-15
**版本：** 2.0 (Clean Workflow)
**状态：** ✅ 推荐使用
```