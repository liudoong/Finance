"""
Equity Volatility Surface - 使用示例
====================================

这个文件展示如何使用 equity_vol_surface.py 模块进行波动率曲面校准

包含的示例:
1. 基本使用 - 手动指定spot price
2. 自动spot price - 从Yahoo Finance下载
3. 导出数据和可视化
4. 比较Heston和SABR模型
5. 使用QuantLib曲面进行期权定价

作者: Claude
日期: 2026-01-10
"""

import sys
import os

# 添加Exotic目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from equity_vol_surface import VolatilitySurfaceCalibrator, calibrate_vol_surface


# =============================================================================
# 示例1: 基本使用 - 手动指定spot price
# =============================================================================

def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("="*80)
    print("示例1: 基本使用 - 手动指定spot price")
    print("="*80)
    print()

    # 创建校准器
    calibrator = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",  # Excel文件路径
        spot_price=5900.0,                             # 手动指定现货价格
        risk_free_rate=0.045,                          # 无风险利率 4.5%
        dividend_yield=0.0,                            # 股息率
        model_type='heston'                            # 模型类型: heston 或 sabr
    )

    # 执行校准
    vol_surface = calibrator.calibrate()

    # 获取校准后的参数
    params = calibrator.get_parameters()

    print("\n校准结果:")
    print(f"  模型类型: {params['model_type']}")
    print(f"  v0 (初始方差): {params['v0']:.6f}")
    print(f"  kappa (均值回归): {params['kappa']:.4f}")
    print(f"  theta (长期方差): {params['theta']:.6f}")
    print(f"  sigma (vol of vol): {params['sigma']:.4f}")
    print(f"  rho (相关系数): {params['rho']:.4f}")

    return calibrator


# =============================================================================
# 示例2: 自动spot price - 从Yahoo Finance下载
# =============================================================================

def example_2_auto_spot_price():
    """示例2: 自动从Yahoo Finance获取spot price"""
    print("\n" + "="*80)
    print("示例2: 自动spot price - 从Yahoo Finance下载")
    print("="*80)
    print()

    try:
        # 使用 'auto' 参数自动下载spot price
        calibrator = VolatilitySurfaceCalibrator(
            option_data_file="spx_infvol_20260109.xlsx",
            spot_price='auto',  # 自动提取ticker并从Yahoo Finance下载
            risk_free_rate=0.045,
            model_type='heston'
        )

        # 执行校准
        vol_surface = calibrator.calibrate()

        print("\n自动获取spot price成功!")
        print(f"使用的ticker: {calibrator.ticker}")
        print(f"数据日期: {calibrator.data_date.strftime('%Y-%m-%d')}")
        print(f"Spot价格: {calibrator.spot:,.2f}")

        return calibrator

    except ImportError:
        print("\n需要安装yfinance: pip install yfinance")
        return None
    except Exception as e:
        print(f"\n自动获取spot price失败: {e}")
        print("请使用手动指定spot_price的方式")
        return None


# =============================================================================
# 示例3: 导出数据和可视化
# =============================================================================

def example_3_export_and_visualize(calibrator):
    """示例3: 导出数据和生成可视化图表"""
    print("\n" + "="*80)
    print("示例3: 导出数据和可视化")
    print("="*80)
    print()

    if calibrator is None:
        print("跳过此示例 (未提供校准器)")
        return

    # 1. 导出曲面数据为CSV
    df = calibrator.export_surface_data(output_file="vol_surface_data.csv")
    print(f"✓ 导出了 {len(df)} 个数据点到 vol_surface_data.csv")
    print()

    # 显示前几行数据
    print("数据预览:")
    print(df.head(10))
    print()

    # 2. 绘制波动率曲面图
    print("生成波动率曲面可视化图...")
    try:
        calibrator.plot_volatility_surface(
            output_file="vol_surface.png",  # 保存为PNG文件
            show_plot=False                  # 设置为True会显示图片窗口
        )
        print("✓ 波动率曲面图已保存到 vol_surface.png")
        print()
        print("生成的图表包含:")
        print("  1. 波动率微笑曲线 (Volatility Smile) - 按期限分组")
        print("  2. 3D波动率曲面 (3D Volatility Surface)")

    except ImportError as e:
        print(f"绘图失败: {e}")
        print("需要安装: pip install matplotlib scipy")


# =============================================================================
# 示例4: 比较Heston和SABR模型
# =============================================================================

def example_4_compare_models():
    """示例4: 比较Heston和SABR模型的校准结果"""
    print("\n" + "="*80)
    print("示例4: 比较Heston和SABR模型")
    print("="*80)
    print()

    # 校准Heston模型
    print("校准Heston模型...")
    heston_cal = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",
        spot_price=5900.0,
        risk_free_rate=0.045,
        model_type='heston'
    )
    heston_surface = heston_cal.calibrate()
    heston_params = heston_cal.get_parameters()

    print("\n" + "-"*80)

    # 校准SABR模型
    print("\n校准SABR模型...")
    sabr_cal = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",
        spot_price=5900.0,
        risk_free_rate=0.045,
        model_type='sabr'
    )
    sabr_surface = sabr_cal.calibrate()
    sabr_params = sabr_cal.get_parameters()

    # 比较结果
    print("\n" + "="*80)
    print("模型比较结果")
    print("="*80)
    print()

    print("Heston模型参数:")
    print(heston_params['parameters'])
    print()

    print("SABR模型参数:")
    print(sabr_params['parameters'])
    print()

    print("模型选择建议:")
    print("  - Heston: 适合长期期权定价、路径依赖衍生品、蒙特卡洛模拟")
    print("  - SABR: 适合短期期权、波动率交易、快速定价")


# =============================================================================
# 示例5: 使用QuantLib曲面进行期权定价
# =============================================================================

def example_5_option_pricing_with_quantlib():
    """示例5: 使用校准后的QuantLib曲面进行期权定价"""
    print("\n" + "="*80)
    print("示例5: 使用QuantLib曲面进行期权定价")
    print("="*80)
    print()

    try:
        import QuantLib as ql
    except ImportError:
        print("需要安装QuantLib: pip install QuantLib-Python")
        return

    # 校准Heston模型
    calibrator = VolatilitySurfaceCalibrator(
        option_data_file="spx_infvol_20260109.xlsx",
        spot_price=5900.0,
        risk_free_rate=0.045,
        model_type='heston'
    )

    heston_model = calibrator.calibrate()

    print("\n使用Heston模型定价欧式期权...")
    print()

    # 设置期权参数
    strike = 5900.0  # ATM期权
    maturity_date = ql.Date(12, 1, 2026)  # 2026年1月12日

    # 创建期权
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    european_option = ql.VanillaOption(payoff, exercise)

    # 使用Heston模型定价
    engine = ql.AnalyticHestonEngine(heston_model)
    european_option.setPricingEngine(engine)

    # 获取定价结果
    price = european_option.NPV()
    delta = european_option.delta()
    gamma = european_option.gamma()
    vega = european_option.vega()

    print(f"欧式看涨期权定价结果:")
    print(f"  现货价格: {calibrator.spot:,.2f}")
    print(f"  行权价: {strike:,.2f}")
    print(f"  到期日: {maturity_date}")
    print(f"  期权价格: ${price:.2f}")
    print(f"  Delta: {delta:.4f}")
    print(f"  Gamma: {gamma:.6f}")
    print(f"  Vega: {vega:.2f}")


# =============================================================================
# 示例6: 便利函数使用
# =============================================================================

def example_6_convenience_function():
    """示例6: 使用便利函数快速校准"""
    print("\n" + "="*80)
    print("示例6: 使用便利函数快速校准")
    print("="*80)
    print()

    # 使用便利函数一行代码完成校准
    vol_surface, params = calibrate_vol_surface(
        option_file="spx_infvol_20260109.xlsx",
        spot_price=5900.0,
        model_type='heston',
        risk_free_rate=0.045
    )

    print("便利函数返回:")
    print(f"  vol_surface: {type(vol_surface)}")
    print(f"  params: {params.keys()}")
    print()
    print("校准完成! 可以直接使用返回的曲面和参数")


# =============================================================================
# 主函数 - 运行所有示例
# =============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "Equity Volatility Surface 使用示例" + " "*23 + "║")
    print("╚" + "═"*78 + "╝")
    print()

    # 运行各个示例
    calibrator_1 = example_1_basic_usage()

    calibrator_2 = example_2_auto_spot_price()

    # 使用第一个校准器进行后续示例
    example_3_export_and_visualize(calibrator_1)

    example_4_compare_models()

    example_5_option_pricing_with_quantlib()

    example_6_convenience_function()

    # 总结
    print("\n" + "="*80)
    print("所有示例运行完成!")
    print("="*80)
    print()
    print("生成的文件:")
    print("  - vol_surface_data.csv  (波动率曲面数据)")
    print("  - vol_surface.png       (波动率曲面可视化图)")
    print()
    print("下一步:")
    print("  1. 查看生成的CSV文件和图片")
    print("  2. 将校准后的参数用于您的定价模型")
    print("  3. 尝试使用不同的Excel数据文件")
    print()


if __name__ == "__main__":
    # 检查是否在Exotic目录下
    if not os.path.exists("spx_infvol_20260109.xlsx"):
        print("错误: 找不到 spx_infvol_20260109.xlsx")
        print("请确保在 Exotic 目录下运行此脚本")
        print()
        print("运行方式:")
        print("  cd Exotic")
        print("  python equity_vol_surface_use.py")
        sys.exit(1)

    # 运行主函数
    main()