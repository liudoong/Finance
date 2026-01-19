# 运行市场 IV 可视化脚本说明

## 📋 前提条件

确保您的 Python 环境已安装以下依赖：
```bash
pip install pandas numpy matplotlib openpyxl scipy
```

## 🚀 运行方法

### 方法 1：直接运行（推荐）

```bash
cd /Users/october/Documents/Github/Finance/Volsurface
python3 plot_market_iv_surface.py
```

### 方法 2：使用特定 Python 环境

如果您有多个 Python 环境，请使用正确的 Python 路径：

```bash
# 查找可用的 Python
which python3

# 或者如果使用 conda/pip 环境
conda activate p311-env  # 或您的环境名
python plot_market_iv_surface.py
```

## 📊 输出结果

脚本会生成：

1. **market_iv_surface.png** - 包含6个子图的综合可视化：
   - 3D 散点图：原始市场 IV 数据
   - 波动率微笑：不同到期日的 IV vs Moneyness
   - ATM 期限结构：ATM 期权的 IV 随时间变化
   - IV 热力图：Moneyness × 到期时间
   - IV 分布直方图：数据分布情况
   - 统计摘要面板：数据质量指标

2. **终端输出**：
   - 数据加载统计
   - 到期日信息
   - IV 范围和分布
   - 数据质量警告（如果有）

## 🎯 这个脚本做什么

**关键特性：**
- ✅ 直接从 Excel 读取 "Implied Volatility" 列
- ✅ 不做任何校准或模型拟合
- ✅ 展示原始市场数据的真实面貌
- ✅ 帮助诊断数据源质量问题

**与校准脚本的区别：**
```
plot_market_iv_surface.py:
  spx_data.xlsx → 读取IV列 → 直接可视化
  （无处理，无校准，无平滑）

clean_workflow.py:
  spx_data.xlsx → 质量采样 → SABR校准 → 插值曲面 → 可视化
  （完整的处理流程）
```

## 🔍 如何使用结果

### 步骤 1：查看原始市场 IV 曲面
运行 `plot_market_iv_surface.py`，打开 `market_iv_surface.png`

**检查：**
- 3D 图是否显示波浪/翻转？
- 期限结构是否有 V 型坑？
- IV 范围是否合理（10%-60%）？
- 是否有极端离群值？

### 步骤 2：对比校准结果
运行 `clean_workflow.py`，打开 `vol_surface_clean_analysis.png`

**对比：**
- 校准后的曲面是否比原始数据更平滑？
- 期限结构是否修复了 V 型坑？
- 波动率范围是否更合理？

### 步骤 3：诊断问题

**场景 A：原始市场 IV 就有波浪**
→ 数据源本身有问题
→ 需要更严格的 `quality_sampler` 参数
→ 或更换数据源

**场景 B：原始市场 IV 平滑，校准后出现波浪**
→ 校准算法有问题
→ 检查 SABR 参数约束
→ 调整正则化强度

**场景 C：原始市场 IV 和校准结果都有波浪**
→ 数据质量极差
→ 使用 `ultra_robust_filter.py` 预过滤
→ 或只保留流动性最好的期权

## ⚙️ 故障排查

### 错误 1: ModuleNotFoundError: No module named 'matplotlib'
```bash
pip install matplotlib
```

### 错误 2: ModuleNotFoundError: No module named 'openpyxl'
```bash
pip install openpyxl
```

### 错误 3: FileNotFoundError: spx_data.xlsx
确保文件在正确位置：
```bash
ls -l /Users/october/Documents/Github/Finance/Volsurface/spx_data.xlsx
```

### 错误 4: 找不到 'Implied Volatility' 列
检查 Excel 文件结构：
```python
import pandas as pd
df = pd.read_excel('spx_data.xlsx', sheet_name=1)
print(df.columns)
```

## 📝 下一步

1. **先运行市场 IV 可视化：**
   ```bash
   python3 plot_market_iv_surface.py
   ```

2. **查看结果：**
   打开 `market_iv_surface.png`，评估原始数据质量

3. **如果原始数据质量好：**
   运行 `clean_workflow.py` 进行校准

4. **如果原始数据质量差：**
   - 调整 `quality_sampler.py` 参数
   - 或使用 `ultra_robust_filter.py` 预处理
   - 或检查数据源是否正确

## 💡 提示

- 这个脚本是**诊断工具**，不是校准工具
- 如果市场 IV 本身就不合理，校准也无法修复
- 始终先检查原始数据，再进行校准
- 对比原始 vs 校准结果，理解算法在做什么

---

**创建日期：** 2026-01-15
**目的：** 可视化原始市场隐含波动率，诊断数据源质量
**相关文件：** plot_market_iv_surface.py, clean_workflow.py