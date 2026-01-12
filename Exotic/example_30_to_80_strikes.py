"""
示例：采样30个strikes，输出80个strikes的高分辨率波动率曲面

这个例子演示了如何：
1. 用少量数据（30 strikes/maturity）进行快速校准
2. 输出高分辨率曲面（80 strikes）
3. 模型通过插值/外推生成额外的strikes
"""

from yahoo_option_data_cleaner import extract_options_data
from data_subsampler import subsample_options_data
from vol_surface_calibrator import calibrate_vol_surface
import time


def example_subsample_30_output_80(excel_file):
    """
    完整示例：采样30个strikes，输出80个strikes

    Args:
        excel_file: Excel文件路径
    """

    print("="*70)
    print("示例：采样30 strikes → 输出80 strikes")
    print("="*70)

    # 步骤1: 提取原始数据
    print("\n[1/4] 提取期权数据...")
    df_raw = extract_options_data(excel_file)
    print(f"✓ 原始数据: {len(df_raw)} 个期权合约")

    # 步骤2: 采样到30个strikes/maturity
    print("\n[2/4] 采样数据（减少到30 strikes/maturity）...")
    df_subsampled = subsample_options_data(
        df_raw,
        num_strikes=30,  # 每个到期日只保留30个期权
        min_options_per_maturity=10
    )
    print(f"✓ 采样后: {len(df_subsampled)} 个期权合约")
    print(f"  数据减少: {len(df_raw)} → {len(df_subsampled)} ({len(df_subsampled)/len(df_raw)*100:.1f}%)")

    # 步骤3: 校准SABR模型，输出80个strikes
    print("\n[3/4] 校准SABR模型（输出80 strikes曲面）...")
    start = time.time()

    result = calibrate_vol_surface(
        df_subsampled,
        model='SABR',
        risk_free_rate=0.05,
        min_maturity_days=7,
        output_strikes=80  # 输出80个strikes的曲面
    )

    elapsed = time.time() - start
    print(f"✓ 校准完成，用时 {elapsed:.1f} 秒")

    # 步骤4: 验证结果
    print("\n[4/4] 验证输出曲面...")
    surface = result['surface']
    params = result['params']

    print(f"\n{'='*70}")
    print("曲面详情")
    print(f"{'='*70}")

    print(f"\n输入数据统计:")
    print(f"  校准数据量: {len(df_subsampled)} 个期权")
    print(f"  平均 strikes/maturity: {len(df_subsampled) / df_subsampled['maturity_date'].nunique():.1f}")

    print(f"\n输出曲面统计:")
    print(f"  行权价数量: {len(surface.strikes)} (目标: 80)")
    print(f"  行权价范围: {surface.strikes.min():.0f} - {surface.strikes.max():.0f}")
    print(f"  行权价间隔: {(surface.strikes.max() - surface.strikes.min()) / (len(surface.strikes) - 1):.2f}")
    print(f"  到期日数量: {len(surface.maturities)}")
    print(f"  曲面大小: {len(surface.strikes)} × {len(surface.maturities)}")

    print(f"\n模型参数:")
    print(f"  SABR模型，每个到期日单独校准")
    print(f"  共 {len(params)} 组参数（每个到期日一组）")

    # 显示前3个到期日的参数
    print(f"\n前3个到期日的SABR参数:")
    for i, (T, p) in enumerate(list(params.items())[:3]):
        print(f"  T={T:.2f}年: α={p['alpha']:.4f}, ρ={p['rho']:.4f}, ν={p['nu']:.4f}, β={p['beta']:.2f}")

    # 测试插值功能
    print(f"\n{'='*70}")
    print("插值测试（查询任意strike的波动率）")
    print(f"{'='*70}")

    spot = result['spot_price']
    test_cases = [
        (spot * 0.90, 0.25, "3个月 90% moneyness"),
        (spot * 0.95, 0.5, "6个月 95% moneyness"),
        (spot, 1.0, "1年 ATM"),
        (spot * 1.05, 1.5, "1.5年 105% moneyness"),
        (spot * 1.10, 2.0, "2年 110% moneyness"),
    ]

    for strike, maturity, desc in test_cases:
        vol = surface.get_vol(strike, maturity)
        print(f"  {desc:25s}: Strike={strike:7.0f}, Vol={vol:.2%}")

    # 导出到QuantLib
    print(f"\n{'='*70}")
    print("导出到QuantLib")
    print(f"{'='*70}")

    try:
        import QuantLib as ql
        ql_surface = surface.to_quantlib()
        print(f"✓ 成功导出到QuantLib BlackVarianceSurface")
        print(f"  可用于Heston/SABR过程、蒙特卡洛模拟等")
    except ImportError:
        print(f"✗ QuantLib未安装，跳过导出")
    except Exception as e:
        print(f"✗ 导出失败: {e}")

    # 保存结果
    print(f"\n{'='*70}")
    print("保存结果")
    print(f"{'='*70}")

    import pandas as pd

    # 保存曲面网格
    grid_data = []
    for i, T in enumerate(surface.maturities):
        for j, K in enumerate(surface.strikes):
            grid_data.append({
                'Strike': K,
                'Maturity_Years': T,
                'Implied_Vol': surface.implied_vols[i, j]
            })

    grid_df = pd.DataFrame(grid_data)
    grid_df.to_csv('vol_surface_30to80.csv', index=False)
    print("✓ 波动率曲面: vol_surface_30to80.csv")

    # 保存参数
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
    params_df.to_csv('sabr_params_30to80.csv', index=False)
    print("✓ SABR参数: sabr_params_30to80.csv")

    print("\n完成！")

    return result


# 主程序
if __name__ == "__main__":

    input_file = "spx_infvol_20260109.xlsx"

    print("\n" + "█"*70)
    print("█ 高分辨率曲面生成示例")
    print("█ 策略：用少量数据校准，输出高分辨率曲面")
    print("█"*70 + "\n")

    result = example_subsample_30_output_80(input_file)

    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print("\n优势:")
    print("  ✓ 校准速度快（只用30个strikes/maturity）")
    print("  ✓ 输出分辨率高（80个strikes）")
    print("  ✓ 模型自动插值/外推到所有strikes")
    print("\n原理:")
    print("  1. SABR/Heston模型是连续函数")
    print("  2. 校准得到模型参数后，可以计算任意(K,T)的波动率")
    print("  3. 输出strikes数量独立于输入数据数量")
    print("\n适用场景:")
    print("  • 需要快速校准但要高分辨率输出")
    print("  • 数据量大但计算资源有限")
    print("  • 需要标准化的高分辨率曲面用于定价")
