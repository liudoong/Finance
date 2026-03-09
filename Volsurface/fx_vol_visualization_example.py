"""
FX波动率曲面可视化示例

展示如何使用 fx_data_generator 生成的数据来可视化波动率曲面
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fx_data_generator import generate_fx_vol_parameters


def visualize_vol_surface_structure(currency_pair="EURUSD"):
    """
    展示波动率曲面的数据结构
    """

    # 生成参数
    df = generate_fx_vol_parameters(currency_pair)

    print(f"\n{'='*80}")
    print(f"波动率曲面数据结构 - {currency_pair}")
    print(f"{'='*80}\n")

    # 期限列表
    tenors = ["1 Week", "1 Month", "2 Month", "3 Month", "4 Month", "5 Month",
              "6 Month", "9 Month", "1 Year", "18 Month", "2 Year", "3 Year",
              "4 Year", "5 Year"]

    # Delta点（波动率微笑的5个点）
    smile_points = ["25P", "10P", "ATM", "10C", "25C"]

    print(f"总共 {len(df)} 条时间序列")
    print(f"- {len(tenors)} 个期限")
    print(f"- 每个期限 9 条数据（5个波动率点 + 4个市场指标）\n")

    print("可视化建议：")
    print("-" * 80)
    print("1. 波动率微笑（2D）：X轴 = Delta, Y轴 = 隐含波动率")
    print("   - 使用前5个点：25P, 10P, ATM, 10C, 25C")
    print("   - 每个期限画一条曲线\n")

    print("2. 期限结构（2D）：X轴 = 期限, Y轴 = 隐含波动率")
    print("   - 选择一个Delta（如ATM）")
    print("   - 跨所有期限画曲线\n")

    print("3. 波动率曲面（3D）：X = Delta, Y = 期限, Z = 隐含波动率")
    print("   - 使用前5个点构建网格")
    print("   - 14个期限 × 5个Delta点 = 70个数据点\n")

    print("=" * 80)

    # 显示数据索引映射
    print("\n数据索引映射（用于提取可视化数据）：")
    print("-" * 80)

    for i, tenor in enumerate(tenors[:3]):  # 只显示前3个作为示例
        start_idx = i * 9
        print(f"\n{tenor}:")
        print(f"  - 波动率微笑点 (可视化用)：行 {start_idx} 到 {start_idx+4}")
        for j, point in enumerate(smile_points):
            row = df.iloc[start_idx + j]
            print(f"    [{start_idx+j:3d}] {point:5s}: {row['expression']}")
        print(f"  - 市场指标 (RR/BF)：     行 {start_idx+5} 到 {start_idx+8}")

    print("\n  ... (其余11个期限同样结构)")
    print("-" * 80)


def create_sample_surface_plot():
    """
    创建示例波动率曲面图（使用模拟数据）
    """

    print("\n生成示例波动率曲面图...")

    # 模拟的隐含波动率数据（实际使用时替换为真实下载的数据）
    deltas = np.array([-25, -10, 0, 10, 25])  # Delta点
    tenors_years = np.array([7/365, 1/12, 2/12, 3/12, 4/12, 5/12,
                             6/12, 9/12, 1, 1.5, 2, 3, 4, 5])  # 期限（年）

    # 创建网格
    Delta, Tenor = np.meshgrid(deltas, tenors_years)

    # 模拟波动率曲面（实际数据来自下载）
    # 经典的波动率微笑：远离ATM的波动率更高，期限越长波动率越平坦
    base_vol = 0.15  # 15% 基础波动率
    smile_effect = 0.03 * (Delta / 25) ** 2  # 微笑效应
    term_structure = 0.05 * (1 - np.exp(-Tenor))  # 期限结构
    Vol = base_vol + smile_effect + term_structure

    # 创建图形
    fig = plt.figure(figsize=(16, 10))

    # 子图1: 3D波动率曲面
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(Delta, Tenor, Vol, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Delta', fontsize=10)
    ax1.set_ylabel('Maturity (Years)', fontsize=10)
    ax1.set_zlabel('Implied Volatility', fontsize=10)
    ax1.set_title('3D Volatility Surface', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 子图2: 波动率微笑（多条曲线 - 不同期限）
    ax2 = fig.add_subplot(2, 2, 2)
    selected_tenors = [0, 3, 7, 9, 13]  # 1W, 3M, 9M, 1Y, 5Y
    tenor_labels = ["1 Week", "3 Month", "9 Month", "1 Year", "5 Year"]

    for idx, tenor_idx in enumerate(selected_tenors):
        ax2.plot(deltas, Vol[tenor_idx, :], marker='o', label=tenor_labels[idx])

    ax2.set_xlabel('Delta', fontsize=10)
    ax2.set_ylabel('Implied Volatility', fontsize=10)
    ax2.set_title('Volatility Smile (Different Maturities)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图3: 期限结构（多条曲线 - 不同Delta）
    ax3 = fig.add_subplot(2, 2, 3)
    selected_deltas = [0, 1, 2, 3, 4]  # -25, -10, 0, 10, 25
    delta_labels = ["25P", "10P", "ATM", "10C", "25C"]

    for idx, delta_idx in enumerate(selected_deltas):
        ax3.plot(tenors_years, Vol[:, delta_idx], marker='o', label=delta_labels[idx])

    ax3.set_xlabel('Maturity (Years)', fontsize=10)
    ax3.set_ylabel('Implied Volatility', fontsize=10)
    ax3.set_title('Term Structure (Different Deltas)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4: 热力图
    ax4 = fig.add_subplot(2, 2, 4)
    im = ax4.imshow(Vol, aspect='auto', cmap='RdYlGn_r',
                    extent=[deltas[0], deltas[-1], tenors_years[-1], tenors_years[0]])
    ax4.set_xlabel('Delta', fontsize=10)
    ax4.set_ylabel('Maturity (Years)', fontsize=10)
    ax4.set_title('Volatility Surface Heatmap', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax4)

    plt.tight_layout()
    plt.savefig('fx_vol_surface_example.png', dpi=150, bbox_inches='tight')
    print("✓ 示例图已保存到: fx_vol_surface_example.png")
    plt.show()


if __name__ == "__main__":
    # 展示数据结构
    visualize_vol_surface_structure("EURUSD")

    # 创建示例可视化
    create_sample_surface_plot()