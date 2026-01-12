"""
波动率曲面校准 - 线性工作流脚本

这是一个逐步执行的脚本，方便理解每个步骤的输入输出。
你可以在Jupyter或IDE中逐块运行，观察每一步的结果。

完整流程：
  Step 1: 提取期权数据
  Step 2: 数据采样（加速校准）
  Step 3: 校准模型参数
  Step 4: 查看波动率曲面
  Step 5: 测试插值功能
  Step 6: 导出到QuantLib
  Step 7: 保存结果到CSV
  Step 8: 可视化波动率曲面
"""

import pandas as pd
import numpy as np
import time
from yahoo_option_data_cleaner import extract_options_data
from data_subsampler import subsample_options_data
from vol_surface_calibrator import calibrate_vol_surface
from plot_vol_surface import plot_volatility_surface, plot_sabr_parameters


# =============================================================================
# 配置参数 - 可以根据需要修改
# =============================================================================

INPUT_FILE = "spx_infvol_20260109.xlsx"  # 输入Excel文件
MODEL = 'SABR'                            # 'SABR' 或 'Heston'
NUM_STRIKES_SUBSAMPLE = 30                # 采样：每个到期日保留多少个期权（控制速度）
NUM_STRIKES_OUTPUT = 80                   # 输出：曲面包含多少个行权价（控制分辨率）
RISK_FREE_RATE = 0.05                     # 无风险利率
MIN_MATURITY_DAYS = 7                     # 过滤掉到期日太近的期权


# =============================================================================
# Step 1: 提取期权数据
# =============================================================================

print("="*80)
print("STEP 1: 提取期权数据")
print("="*80)
print(f"\n从文件读取: {INPUT_FILE}")
print("提取期权合约数据和现货价格...\n")

start_time = time.time()
df_raw = extract_options_data(INPUT_FILE)
elapsed = time.time() - start_time

print(f"\n✓ 数据提取完成 (用时 {elapsed:.1f} 秒)")
print(f"\n数据概览:")
print(f"  总期权数量: {len(df_raw)}")
print(f"  列名: {list(df_raw.columns)}")
print(f"  到期日数量: {df_raw['maturity_date'].nunique()}")
print(f"  行权价范围: {df_raw['strike'].min():.0f} - {df_raw['strike'].max():.0f}")
print(f"  现货价格: {df_raw['spot_price'].iloc[0]:.2f}")

print(f"\n前5行数据:")
print(df_raw.head())

# 暂停点 - 你可以在这里检查 df_raw
# print("\n按回车继续...")
# input()


# =============================================================================
# Step 2: 智能采样数据（加速校准）
# =============================================================================

print("\n\n" + "="*80)
print("STEP 2: 智能采样数据")
print("="*80)
print(f"\n原始数据有 {len(df_raw)} 个期权，校准会很慢")
print(f"目标：每个到期日保留 {NUM_STRIKES_SUBSAMPLE} 个期权")
print("策略：优先保留ATM期权，同时覆盖广泛的moneyness范围\n")

start_time = time.time()
df_subsampled = subsample_options_data(
    df_raw,
    num_strikes=NUM_STRIKES_SUBSAMPLE,
    min_options_per_maturity=10
)
elapsed = time.time() - start_time

print(f"\n✓ 采样完成 (用时 {elapsed:.1f} 秒)")
print(f"\n采样结果:")
print(f"  原始数据: {len(df_raw)} 个期权")
print(f"  采样后: {len(df_subsampled)} 个期权 ({len(df_subsampled)/len(df_raw)*100:.1f}%)")
print(f"  数据减少: {len(df_raw) - len(df_subsampled)} 个期权")
print(f"  平均 strikes/maturity: {len(df_subsampled) / df_subsampled['maturity_date'].nunique():.1f}")

print(f"\n每个到期日的期权数量:")
maturity_counts = df_subsampled['maturity_date'].value_counts().sort_index()
for mat, count in list(maturity_counts.items())[:5]:
    print(f"  {mat}: {count} 个期权")
if len(maturity_counts) > 5:
    print(f"  ... 共 {len(maturity_counts)} 个到期日")

# 暂停点 - 你可以在这里检查 df_subsampled
# print("\n按回车继续...")
# input()


# =============================================================================
# Step 3: 校准模型
# =============================================================================

print("\n\n" + "="*80)
print("STEP 3: 校准波动率模型")
print("="*80)
print(f"\n模型: {MODEL}")
print(f"校准数据: {len(df_subsampled)} 个期权")
print(f"输出曲面: {NUM_STRIKES_OUTPUT} 个行权价")
print("\n开始校准...\n")

start_time = time.time()
result = calibrate_vol_surface(
    df_subsampled,
    model=MODEL,
    risk_free_rate=RISK_FREE_RATE,
    min_maturity_days=MIN_MATURITY_DAYS,
    output_strikes=NUM_STRIKES_OUTPUT
)
calibration_time = time.time() - start_time

print(f"\n✓ 校准完成 (用时 {calibration_time:.1f} 秒)")

# 提取结果
surface = result['surface']
params = result['params']
spot = result['spot_price']

print(f"\n校准结果摘要:")
print(f"  模型: {MODEL}")
print(f"  校准时间: {calibration_time:.1f} 秒")
print(f"  现货价格: {spot:.2f}")

# 显示模型参数
if MODEL == 'SABR':
    print(f"\n  SABR参数 (每个到期日单独校准):")
    print(f"  共 {len(params)} 组参数")
    print(f"\n  前5个到期日的参数:")
    for T, p in list(params.items())[:5]:
        print(f"    T={T:.2f}年: α={p['alpha']:.4f}, β={p['beta']:.2f}, ρ={p['rho']:.4f}, ν={p['nu']:.4f}")
    if len(params) > 5:
        print(f"    ... (共 {len(params)} 个到期日)")
else:  # Heston
    print(f"\n  Heston参数 (全局参数):")
    for key, value in params.items():
        if key != 'calibration_error':
            print(f"    {key}: {value:.6f}")

# 暂停点
# print("\n按回车继续...")
# input()


# =============================================================================
# Step 4: 检查波动率曲面
# =============================================================================

print("\n\n" + "="*80)
print("STEP 4: 波动率曲面详情")
print("="*80)

print(f"\n曲面维度:")
print(f"  行权价数量: {len(surface.strikes)}")
print(f"  到期日数量: {len(surface.maturities)}")
print(f"  曲面大小: {len(surface.strikes)} × {len(surface.maturities)}")

print(f"\n行权价范围:")
print(f"  最小: {surface.strikes.min():.0f}")
print(f"  最大: {surface.strikes.max():.0f}")
print(f"  间隔: {(surface.strikes.max() - surface.strikes.min()) / (len(surface.strikes) - 1):.2f}")

print(f"\n到期日范围:")
print(f"  最短: {surface.maturities.min():.2f} 年")
print(f"  最长: {surface.maturities.max():.2f} 年")

print(f"\n波动率范围:")
print(f"  最小波动率: {surface.implied_vols.min():.2%}")
print(f"  最大波动率: {surface.implied_vols.max():.2%}")
print(f"  平均波动率: {surface.implied_vols.mean():.2%}")

# 显示前5个行权价
print(f"\n前5个行权价: {surface.strikes[:5]}")
print(f"前5个到期日: {surface.maturities[:5]}")

# 显示波动率矩阵的一角
print(f"\n波动率矩阵 (前3×3):")
print(surface.implied_vols[:3, :3])

# 暂停点
# print("\n按回车继续...")
# input()


# =============================================================================
# Step 5: 测试插值功能
# =============================================================================

print("\n\n" + "="*80)
print("STEP 5: 测试波动率插值")
print("="*80)
print("\n曲面支持查询任意 (strike, maturity) 的波动率")
print("即使这些点不在原始数据中，模型也能插值/外推\n")

test_cases = [
    (spot * 0.90, 0.25, "3个月 OTM Put (90% moneyness)"),
    (spot * 0.95, 0.5, "6个月 OTM Put (95% moneyness)"),
    (spot, 0.5, "6个月 ATM"),
    (spot, 1.0, "1年 ATM"),
    (spot * 1.05, 1.0, "1年 OTM Call (105% moneyness)"),
    (spot * 1.10, 2.0, "2年 OTM Call (110% moneyness)"),
]

print(f"现货价格: {spot:.2f}\n")
print(f"{'描述':<35} {'Strike':<10} {'Maturity':<10} {'波动率':<10}")
print("-" * 80)

for strike, maturity, desc in test_cases:
    vol = surface.get_vol(strike, maturity)
    print(f"{desc:<35} {strike:<10.0f} {maturity:<10.2f} {vol:<10.2%}")

print("\n说明：这些波动率是模型根据校准参数计算的，不是直接从市场数据读取的")

# 暂停点
# print("\n按回车继续...")
# input()


# =============================================================================
# Step 6: 导出到QuantLib
# =============================================================================

print("\n\n" + "="*80)
print("STEP 6: 导出到QuantLib")
print("="*80)
print("\nQuantLib BlackVarianceSurface可用于:")
print("  • Heston/SABR随机过程")
print("  • 蒙特卡洛模拟")
print("  • PDE定价引擎")
print("  • 奇异期权定价\n")

try:
    import QuantLib as ql
    ql_surface = surface.to_quantlib()
    print("✓ 成功导出到QuantLib BlackVarianceSurface")
    print(f"  类型: {type(ql_surface)}")
    print(f"  可直接用于QuantLib的定价引擎")

    # 测试从QuantLib曲面查询波动率
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()
    today = ql.Date.todaysDate()

    test_date = today + ql.Period(180, ql.Days)  # 6个月后
    test_strike = spot

    variance = ql_surface.blackVariance(test_date, test_strike)
    vol_from_ql = np.sqrt(variance / 0.5)  # 转换为波动率

    print(f"\n  测试查询 (6个月ATM):")
    print(f"    从QuantLib曲面获得的波动率: {vol_from_ql:.2%}")

except ImportError:
    print("✗ QuantLib未安装")
    print("  安装命令: pip install QuantLib-Python")
except Exception as e:
    print(f"✗ 导出失败: {e}")

# 暂停点
# print("\n按回车继续...")
# input()


# =============================================================================
# Step 7: 保存结果到文件
# =============================================================================

print("\n\n" + "="*80)
print("STEP 7: 保存结果")
print("="*80)

# 1. 保存波动率曲面网格
print("\n[1/3] 保存波动率曲面网格...")
grid_data = []
for i, T in enumerate(surface.maturities):
    for j, K in enumerate(surface.strikes):
        grid_data.append({
            'Strike': K,
            'Maturity_Years': T,
            'Maturity_Days': int(T * 365),
            'Implied_Vol': surface.implied_vols[i, j],
            'Moneyness': K / spot
        })

grid_df = pd.DataFrame(grid_data)
grid_df.to_csv('vol_surface_grid.csv', index=False)
print(f"  ✓ 波动率曲面网格: vol_surface_grid.csv")
print(f"    {len(grid_df)} 个数据点 ({len(surface.strikes)} strikes × {len(surface.maturities)} maturities)")

# 2. 保存模型参数
print("\n[2/3] 保存模型参数...")
if MODEL == 'SABR':
    params_list = []
    for T, p in params.items():
        params_list.append({
            'Maturity_Years': T,
            'Maturity_Days': int(T * 365),
            'alpha': p['alpha'],
            'beta': p['beta'],
            'rho': p['rho'],
            'nu': p['nu']
        })
    params_df = pd.DataFrame(params_list)
    params_df.to_csv('calibrated_params.csv', index=False)
    print(f"  ✓ SABR参数: calibrated_params.csv")
    print(f"    {len(params_df)} 组参数（每个到期日一组）")
else:  # Heston
    params_df = pd.DataFrame([{
        'Parameter': key,
        'Value': value
    } for key, value in params.items()])
    params_df.to_csv('calibrated_params.csv', index=False)
    print(f"  ✓ Heston参数: calibrated_params.csv")
    print(f"    {len(params_df)} 个参数")

# 3. 保存采样后的期权数据
print("\n[3/3] 保存采样后的期权数据...")
df_subsampled.to_csv('options_subsampled.csv', index=False)
print(f"  ✓ 采样期权数据: options_subsampled.csv")
print(f"    {len(df_subsampled)} 个期权合约")


# =============================================================================
# Step 8: 可视化波动率曲面
# =============================================================================

print("\n\n" + "="*80)
print("STEP 8: 绘制波动率曲面图表")
print("="*80)
print("\n生成可视化图表...")
print("包含: 3D曲面、波动率微笑、期限结构、热图、参数统计\n")

try:
    # 绘制完整的波动率曲面分析图
    plot_volatility_surface(
        result,
        save_path=f'vol_surface_{MODEL.lower()}_analysis.png',
        show=False  # 设置为False避免阻塞，图片会保存
    )
    print(f"✓ 波动率曲面分析图已保存: vol_surface_{MODEL.lower()}_analysis.png")

    # 如果是SABR模型，额外绘制参数演化图
    if MODEL == 'SABR':
        plot_sabr_parameters(
            result,
            save_path=f'sabr_parameters_{MODEL.lower()}.png',
            show=False
        )
        print(f"✓ SABR参数演化图已保存: sabr_parameters_{MODEL.lower()}.png")

    print("\n提示: 可以打开图片查看，或者将 show=False 改为 show=True 直接显示")

except ImportError as e:
    print(f"✗ 绘图失败: matplotlib未安装")
    print(f"  安装命令: pip install matplotlib")
except Exception as e:
    print(f"✗ 绘图失败: {e}")

# 暂停点
# print("\n按回车继续...")
# input()


# =============================================================================
# 总结
# =============================================================================

print("\n\n" + "="*80)
print("完成！工作流总结")
print("="*80)

print(f"""
数据流转:
  {len(df_raw)} 个原始期权
      ↓ (智能采样)
  {len(df_subsampled)} 个期权用于校准
      ↓ ({MODEL}模型校准)
  {len(surface.strikes)} × {len(surface.maturities)} 波动率曲面

时间:
  数据提取: {elapsed:.1f} 秒
  校准时间: {calibration_time:.1f} 秒

输出文件:
  • vol_surface_grid.csv - 波动率曲面网格 ({len(grid_df)} 个点)
  • calibrated_params.csv - 模型参数
  • options_subsampled.csv - 用于校准的期权数据
  • vol_surface_{MODEL.lower()}_analysis.png - 波动率曲面可视化图表
  {'• sabr_parameters_' + MODEL.lower() + '.png - SABR参数演化图' if MODEL == 'SABR' else ''}

下一步:
  1. 打开 vol_surface_{MODEL.lower()}_analysis.png 查看完整的曲面分析
  2. 在Excel中打开 vol_surface_grid.csv 查看数据
  3. 用波动率曲面定价奇异期权
  4. 将QuantLib曲面用于蒙特卡洛模拟
  5. 比较不同模型的校准结果
""")

print("="*80)