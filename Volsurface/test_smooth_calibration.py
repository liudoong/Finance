"""
测试脚本：对比原始校准 vs 平滑校准

运行此脚本对比：
1. 原始 SABR 校准（可能有 bumps）
2. 平滑 SABR 校准（regularization=0.01）
3. 强平滑 SABR 校准（regularization=0.05）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from option_data_cleaner import extract_options_data
from data_subsampler import subsample_options_data
from vol_surface_calibrator import calibrate_vol_surface
from smooth_calibrator import smooth_calibrate_vol_surface

# 配置
INPUT_FILE = "spx_data.xlsx"
USE_SUBSAMPLING = True  # 是否先子采样
NUM_STRIKES_SUBSAMPLE = 50


def plot_term_structure_comparison(results_dict, spot):
    """
    绘制 ATM 波动率期限结构对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ATM 期限结构
    ax1 = axes[0]
    maturities = np.linspace(0.1, 2.0, 30)

    for name, result in results_dict.items():
        surface = result['surface']
        atm_vols = [surface.get_vol(spot, T) for T in maturities]
        ax1.plot(maturities, atm_vols, 'o-', label=name, markersize=3, alpha=0.7)

    ax1.set_xlabel('Maturity (years)', fontsize=12)
    ax1.set_ylabel('ATM Implied Volatility', fontsize=12)
    ax1.set_title('ATM Term Structure Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1年期 volatility smile
    ax2 = axes[1]
    test_maturity = 1.0
    strikes = np.linspace(spot * 0.8, spot * 1.2, 50)
    moneyness = strikes / spot

    for name, result in results_dict.items():
        surface = result['surface']
        vols = [surface.get_vol(K, test_maturity) for K in strikes]
        ax2.plot(moneyness, vols, '-', label=name, linewidth=2, alpha=0.7)

    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax2.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax2.set_ylabel('Implied Volatility', fontsize=12)
    ax2.set_title(f'Volatility Smile (T={test_maturity}y)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('smooth_calibration_comparison.png', dpi=150)
    print("\n✓ 对比图已保存: smooth_calibration_comparison.png")
    plt.show()


def plot_parameter_continuity(params_dict, param_name='alpha'):
    """
    绘制参数连续性对比
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, params in params_dict.items():
        if isinstance(params, dict) and all(isinstance(v, dict) for v in params.values()):
            # SABR 参数（按 maturity）
            maturities = sorted(params.keys())
            values = [params[T][param_name] for T in maturities]

            ax.plot(maturities, values, 'o-', label=name, markersize=6, linewidth=2, alpha=0.7)

    ax.set_xlabel('Maturity (years)', fontsize=12)
    ax.set_ylabel(f'SABR Parameter: {param_name}', fontsize=12)
    ax.set_title(f'Parameter Continuity: {param_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'parameter_{param_name}_continuity.png', dpi=150)
    print(f"✓ 参数图已保存: parameter_{param_name}_continuity.png")
    plt.show()


def main():
    print("="*80)
    print(" 波动率曲面平滑校准对比测试")
    print("="*80)

    # Step 1: 加载数据
    print("\n[1/5] 加载数据...")
    df = extract_options_data(INPUT_FILE)
    print(f"    加载了 {len(df)} 个期权合约")

    # Step 2: 子采样（可选）
    if USE_SUBSAMPLING:
        print(f"\n[2/5] 子采样到 {NUM_STRIKES_SUBSAMPLE} strikes/maturity...")
        df = subsample_options_data(df, num_strikes=NUM_STRIKES_SUBSAMPLE)
        print(f"    子采样后: {len(df)} 个期权")
    else:
        print("\n[2/5] 跳过子采样（使用全部数据）")

    spot = df['spot_price'].iloc[0]

    # Step 3: 原始校准
    print("\n[3/5] 原始 SABR 校准（无正则化）...")
    print("-" * 80)
    result_original = calibrate_vol_surface(
        df,
        model='SABR',
        output_strikes=50
    )

    # Step 4: 平滑校准（中度）
    print("\n[4/5] 平滑 SABR 校准（regularization=0.01）...")
    print("-" * 80)
    result_smooth_medium = smooth_calibrate_vol_surface(
        df,
        model='SABR',
        regularization=0.01,
        filter_data=True,
        outlier_threshold=2.5,
        output_strikes=50
    )

    # Step 5: 平滑校准（强）
    print("\n[5/5] 强平滑 SABR 校准（regularization=0.05）...")
    print("-" * 80)
    result_smooth_strong = smooth_calibrate_vol_surface(
        df,
        model='SABR',
        regularization=0.05,
        filter_data=True,
        outlier_threshold=2.5,
        output_strikes=50
    )

    # 汇总结果
    results = {
        'Original (no reg)': result_original,
        'Smooth (λ=0.01)': result_smooth_medium,
        'Strong Smooth (λ=0.05)': result_smooth_strong
    }

    params_dict = {
        'Original': result_original['params'],
        'Smooth (λ=0.01)': result_smooth_medium['params'],
        'Strong Smooth (λ=0.05)': result_smooth_strong['params']
    }

    # 对比分析
    print("\n" + "="*80)
    print(" 对比分析")
    print("="*80)

    # ATM vol 对比
    print("\nATM Volatility Comparison (T=1.0y, K=Spot):")
    print("-" * 50)
    test_maturity = 1.0
    for name, result in results.items():
        surface = result['surface']
        atm_vol = surface.get_vol(spot, test_maturity)
        print(f"  {name:25s}: {atm_vol:.2%}")

    # 参数跳跃分析
    print("\n\nParameter Jump Analysis (max |Δα| between consecutive maturities):")
    print("-" * 50)
    for name, params in params_dict.items():
        if isinstance(params, dict) and all(isinstance(v, dict) for v in params.values()):
            maturities = sorted(params.keys())
            alphas = [params[T]['alpha'] for T in maturities]

            max_jump = 0
            for i in range(1, len(alphas)):
                jump = abs(alphas[i] - alphas[i-1])
                max_jump = max(max_jump, jump)

            print(f"  {name:25s}: {max_jump:.4f}")

    # 绘图
    print("\n" + "="*80)
    print(" 生成对比图...")
    print("="*80)

    plot_term_structure_comparison(results, spot)
    plot_parameter_continuity(params_dict, param_name='alpha')
    plot_parameter_continuity(params_dict, param_name='rho')
    plot_parameter_continuity(params_dict, param_name='nu')

    # 推荐
    print("\n" + "="*80)
    print(" 推荐")
    print("="*80)
    print("""
    观察对比图后：

    ✓ 如果原始校准曲面已经平滑 → 使用原始方法
    ✓ 如果看到明显 bumps/颠簸 → 使用 λ=0.01 平滑校准
    ✓ 如果仍有小颠簸 → 使用 λ=0.05 强平滑
    ✓ 如果过度平滑、丢失 smile → 降低 λ 到 0.005

    关键指标：
    - ATM term structure 应该平滑单调
    - Parameter jumps 应该较小（<0.05）
    - Smile 形状应该合理（中间高、两边低）
    """)

    print("\n✓ 测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()
