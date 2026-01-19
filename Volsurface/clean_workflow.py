"""
清洁的波动率曲面校准工作流

核心改进：
1. 使用 quality_sampler 在采样阶段就过滤坏数据
2. 校准时不再需要额外过滤（数据已经很干净）
3. 简化流程，专注于质量

这个版本替代 fast_calibration_workflow.py，更简洁、更可靠
"""

import pandas as pd
import numpy as np
import time
from option_data_cleaner import extract_options_data
from quality_sampler import quality_subsample_options
from vol_surface_calibrator import calibrate_vol_surface
from plot_vol_surface import plot_volatility_surface, plot_sabr_parameters

# =============================================================================
# 配置参数
# =============================================================================

INPUT_FILE = "spx_data.xlsx"
MODEL = 'Heston'

# 质量采样参数
NUM_STRIKES_PER_MATURITY = 40    # 每个到期日的目标期权数
MIN_TIME_VALUE_PCT = 0.03        # 最小时间价值 3%
MONEYNESS_RANGE = (0.85, 1.20)   # 只保留接近 ATM
MIN_IV = 0.10                    # 最小 IV 10%
MAX_IV = 0.60                    # 最大 IV 60%
MIN_OPTIONS_PER_MATURITY = 20    # 每个到期日最少期权数（否则放弃该到期日）

# 校准参数
NUM_STRIKES_OUTPUT = 80
RISK_FREE_RATE = 0.0375


# =============================================================================
# Step 1: 提取原始数据
# =============================================================================

print("\n" + "="*80)
print("Step 1: 提取原始数据")
print("="*80)

df_raw = extract_options_data(INPUT_FILE)

print(f"\n原始数据统计:")
print(f"  总期权数: {len(df_raw)}")
print(f"  到期日数: {df_raw['maturity_date'].nunique()}")
print(f"  行权价范围: {df_raw['strike'].min():.0f} - {df_raw['strike'].max():.0f}")
print(f"  现货价格: {df_raw['spot_price'].iloc[0]:.2f}")


# =============================================================================
# Step 2: 质量优先采样（采样+过滤一体化）
# =============================================================================

print("\n" + "="*80)
print("Step 2: 质量优先采样")
print("="*80)
print(f"策略:")
print(f"  - 时间价值: >={MIN_TIME_VALUE_PCT*100:.0f}%")
print(f"  - Moneyness: {MONEYNESS_RANGE[0]} - {MONEYNESS_RANGE[1]} (聚焦ATM)")
print(f"  - IV 范围: {MIN_IV*100:.0f}% - {MAX_IV*100:.0f}%")
print(f"  - 每到期日: {NUM_STRIKES_PER_MATURITY} 个高质量期权")

start_time = time.time()

df_quality = quality_subsample_options(
    df_raw,
    num_strikes_per_maturity=NUM_STRIKES_PER_MATURITY,
    min_time_value_pct=MIN_TIME_VALUE_PCT,
    moneyness_range=MONEYNESS_RANGE,
    min_iv=MIN_IV,
    max_iv=MAX_IV,
    min_options_per_maturity=MIN_OPTIONS_PER_MATURITY,
    risk_free_rate=RISK_FREE_RATE
)

sampling_time = time.time() - start_time

print(f"\n✓ 质量采样完成 (用时 {sampling_time:.1f} 秒)")


# =============================================================================
# Step 3: 校准模型（使用原始校准器，数据已经很干净）
# =============================================================================

print("\n" + "="*80)
print("Step 3: SABR 模型校准")
print("="*80)
print(f"使用 {len(df_quality)} 个高质量期权")
print(f"数据已预过滤，直接校准...\n")

start_time = time.time()

result = calibrate_vol_surface(
    df_quality,
    model=MODEL,
    risk_free_rate=RISK_FREE_RATE,
    min_maturity_days=7,
    output_strikes=NUM_STRIKES_OUTPUT
)

calibration_time = time.time() - start_time

print(f"\n✓ 校准完成 (用时 {calibration_time:.1f} 秒)")


# 提取结果
surface = result['surface']
params = result['params']
spot = result['spot_price']


# =============================================================================
# Step 4: 结果分析
# =============================================================================

print("\n" + "="*80)
print("Step 4: 校准结果分析")
print("="*80)

print(f"\n模型参数:")
print(f"  模型: {MODEL}")
print(f"  现货价格: ${spot:.2f}")
print(f"  到期日数: {len(params)}")

# SABR 参数
print(f"\n前5个到期日的SABR参数:")
for T, p in list(params.items())[:5]:
    print(f"  T={T:.2f}年: α={p['alpha']:.4f}, β={p['beta']:.2f}, ρ={p['rho']:+.4f}, ν={p['nu']:.4f}")

if len(params) > 5:
    print(f"  ... (共 {len(params)} 个到期日)")

# 参数连续性检查
if len(params) > 1:
    maturities = sorted(params.keys())
    alphas = [params[T]['alpha'] for T in maturities]
    max_alpha_jump = max([abs(alphas[i+1] - alphas[i]) for i in range(len(alphas)-1)])

    print(f"\n参数连续性检查:")
    print(f"  最大α跳跃: {max_alpha_jump:.4f}")

    if max_alpha_jump < 0.05:
        print(f"  ✓ 参数平滑，曲面应该很好")
    elif max_alpha_jump < 0.10:
        print(f"  ⚠ 参数有轻微跳跃，曲面可能有小波动")
    else:
        print(f"  ⚠⚠ 参数跳跃较大，检查数据质量")

# 曲面统计
print(f"\n波动率曲面:")
print(f"  最小波动率: {surface.implied_vols.min()*100:.1f}%")
print(f"  最大波动率: {surface.implied_vols.max()*100:.1f}%")
print(f"  平均波动率: {surface.implied_vols.mean()*100:.1f}%")
print(f"  曲面大小: {len(surface.strikes)} × {len(surface.maturities)}")


# =============================================================================
# Step 5: 测试插值
# =============================================================================

print("\n" + "="*80)
print("Step 5: 测试波动率插值")
print("="*80)

test_cases = [
    (spot * 0.90, 0.25, "3个月 10% OTM Put"),
    (spot, 0.5, "6个月 ATM"),
    (spot, 1.0, "1年 ATM"),
    (spot * 1.05, 1.0, "1年 5% OTM Call"),
]

print(f"\n现货价格: {spot:.2f}\n")
for strike, maturity, desc in test_cases:
    vol = surface.get_vol(strike, maturity)
    print(f"{desc:<25} K={strike:<8.0f} T={maturity:.2f}y  →  IV = {vol:.2%}")


# =============================================================================
# Step 6: 保存结果
# =============================================================================

print("\n" + "="*80)
print("Step 6: 保存结果")
print("="*80)

# 1. 质量采样的数据
df_quality.to_csv('options_quality_sampled.csv', index=False)
print(f"✓ 质量数据: options_quality_sampled.csv")

# 2. 波动率曲面网格
grid_data = []
for i, T in enumerate(surface.maturities):
    for j, K in enumerate(surface.strikes):
        grid_data.append({
            'Strike': K,
            'Maturity_Years': T,
            'Implied_Vol': surface.implied_vols[i, j],
            'Moneyness': K / spot
        })

grid_df = pd.DataFrame(grid_data)
grid_df.to_csv('vol_surface_grid_clean.csv', index=False)
print(f"✓ 波动率网格: vol_surface_grid_clean.csv ({len(grid_df)} 个点)")

# 3. SABR 参数
params_list = []
for T, p in params.items():
    params_list.append({
        'Maturity_Years': T,
        'alpha': p['alpha'],
        'beta': p['beta'],
        'rho': p['rho'],
        'nu': p['nu']
    })

params_df = pd.DataFrame(params_list)
params_df.to_csv('sabr_params_clean.csv', index=False)
print(f"✓ SABR参数: sabr_params_clean.csv ({len(params_df)} 组)")


# =============================================================================
# Step 7: 可视化
# =============================================================================

print("\n" + "="*80)
print("Step 7: 生成可视化图表")
print("="*80)

try:
    plot_volatility_surface(
        result,
        save_path='vol_surface_clean_analysis.png',
        show=True
    )
    print(f"✓ 波动率曲面分析图: vol_surface_clean_analysis.png")

    plot_sabr_parameters(
        result,
        save_path='sabr_params_clean.png',
        show=True
    )
    print(f"✓ SABR参数演化图: sabr_params_clean.png")

except Exception as e:
    print(f"⚠ 绘图失败: {e}")


# =============================================================================
# 总结
# =============================================================================

print("\n" + "="*80)
print("完成！总结")
print("="*80)

print(f"""
数据流转:
  {len(df_raw)} 个原始期权
      ↓ (质量优先采样)
  {len(df_quality)} 个高质量期权 ({len(df_quality)/len(df_raw)*100:.1f}%)
      ↓ (SABR校准)
  {len(surface.strikes)} × {len(surface.maturities)} 波动率曲面

时间:
  质量采样: {sampling_time:.1f} 秒
  模型校准: {calibration_time:.1f} 秒

数据质量:
  IV范围: {df_quality['implied_vol'].min()*100:.1f}% - {df_quality['implied_vol'].max()*100:.1f}%
  参数跳跃: {max_alpha_jump:.4f} {'✓' if max_alpha_jump < 0.05 else '⚠'}

输出文件:
  • options_quality_sampled.csv - 高质量期权数据
  • vol_surface_grid_clean.csv - 波动率曲面网格
  • sabr_params_clean.csv - SABR参数
  • vol_surface_clean_analysis.png - 曲面分析图
  • sabr_params_clean.png - 参数演化图

下一步:
  1. 打开 vol_surface_clean_analysis.png 检查曲面
  2. 确认 Term Structure 平滑单调
  3. 检查 Smile 形状合理
  4. 如果仍有问题，调整采样参数:
     - 收紧 MONEYNESS_RANGE (例如 0.90-1.15)
     - 降低 MAX_IV (例如 0.50)
     - 增加 MIN_TIME_VALUE_PCT (例如 0.04)
""")

print("="*80)