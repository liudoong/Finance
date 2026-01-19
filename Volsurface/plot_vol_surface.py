"""
波动率曲面可视化工具

提供多种图表展示校准后的波动率曲面:
1. 3D曲面图
2. 波动率微笑（不同到期日）
3. 期限结构（不同行权价）
4. 热图
5. 模型参数可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import gridspec


def plot_volatility_surface(result, save_path=None, show=True):
    """
    绘制完整的波动率曲面分析图表

    Args:
        result: calibrate_vol_surface() 的输出字典，包含:
                - 'surface': VolatilitySurface对象
                - 'params': 模型参数
                - 'model': 模型名称 ('HESTON' 或 'SABR')
                - 'spot_price': 现货价格
        save_path: 保存图片的路径（可选）
        show: 是否显示图表（默认True）
    """

    surface = result['surface']
    params = result['params']
    model = result['model']
    spot = result['spot_price']

    # 创建大图，包含多个子图
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 标题
    fig.suptitle(f'{model} Model - Volatility Surface Analysis\nSpot Price: {spot:.2f}',
                 fontsize=16, fontweight='bold')

    # =========================================================================
    # 1. 3D曲面图
    # =========================================================================
    ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')

    K_grid = np.linspace(surface.strikes.min(), surface.strikes.max(), 60)
    T_grid = np.linspace(surface.maturities.min(), surface.maturities.max(), 60)
    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)

    V_mesh = np.zeros_like(K_mesh)
    for i in range(len(T_grid)):
        # Ensure get_vol returns the correct shape
        vols = surface.get_vol(K_grid, np.full(len(K_grid), T_grid[i]))
        if np.isscalar(vols):
            V_mesh[i, :] = vols
        else:
            V_mesh[i, :] = vols

    surf = ax1.plot_surface(K_mesh, T_mesh, V_mesh,
                            cmap='viridis', alpha=0.8,
                            edgecolor='none', antialiased=True)

    ax1.set_xlabel('Strike Price', fontsize=10, labelpad=10)
    ax1.set_ylabel('Maturity (years)', fontsize=10, labelpad=10)
    ax1.set_zlabel('Implied Volatility', fontsize=10, labelpad=10)
    ax1.set_title('3D Volatility Surface', fontsize=12, fontweight='bold')

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    cbar.set_label('Volatility', fontsize=9)

    # 设置视角
    ax1.view_init(elev=25, azim=135)

    # =========================================================================
    # 2. 波动率微笑（不同到期日）
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    # 选择几个代表性的到期日
    maturities_to_plot = []
    for T_target in [0.25, 0.5, 1.0, 2.0, 3.0]:
        if surface.maturities.min() <= T_target <= surface.maturities.max():
            maturities_to_plot.append(T_target)

    # 如果没有合适的，就用实际的前5个
    if len(maturities_to_plot) == 0:
        maturities_to_plot = sorted(surface.maturities)[:5]

    colors = cm.plasma(np.linspace(0, 0.9, len(maturities_to_plot)))

    for T, color in zip(maturities_to_plot, colors):
        vols = surface.get_vol(K_grid, np.full(len(K_grid), T))
        moneyness = K_grid / spot
        ax2.plot(moneyness, vols, label=f'T={T:.2f}y',
                linewidth=2, color=color)

    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5,
                linewidth=1.5, label='ATM')
    ax2.set_xlabel('Moneyness (K/S)', fontsize=10)
    ax2.set_ylabel('Implied Volatility', fontsize=10)
    ax2.set_title('Volatility Smile', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # 3. 期限结构（Term Structure）
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 2])

    # 选择几个代表性的行权价
    moneyness_levels = [0.90, 0.95, 1.00, 1.05, 1.10]
    strikes_to_plot = [spot * m for m in moneyness_levels
                      if surface.strikes.min() <= spot * m <= surface.strikes.max()]

    if len(strikes_to_plot) == 0:
        strikes_to_plot = [surface.strikes[i] for i in
                          np.linspace(0, len(surface.strikes)-1, 5, dtype=int)]

    colors = cm.coolwarm(np.linspace(0, 1, len(strikes_to_plot)))

    T_grid_term = np.linspace(surface.maturities.min(),
                              surface.maturities.max(), 50)

    for K, color in zip(strikes_to_plot, colors):
        vols = surface.get_vol(np.full(len(T_grid_term), K), T_grid_term)
        m = K / spot
        ax3.plot(T_grid_term, vols,
                label=f'K={K:.0f} ({m:.2f})',
                linewidth=2, color=color)

    ax3.set_xlabel('Maturity (years)', fontsize=10)
    ax3.set_ylabel('Implied Volatility', fontsize=10)
    ax3.set_title('Term Structure', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # 4. 热图（Heatmap）
    # =========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    # 使用实际的曲面数据
    moneyness_grid = surface.strikes / spot

    im = ax4.imshow(surface.implied_vols, aspect='auto',
                    cmap='RdYlGn_r', interpolation='bilinear',
                    extent=[moneyness_grid.min(), moneyness_grid.max(),
                           surface.maturities.max(), surface.maturities.min()])

    ax4.set_xlabel('Moneyness (K/S)', fontsize=10)
    ax4.set_ylabel('Maturity (years)', fontsize=10)
    ax4.set_title('Volatility Heatmap', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Volatility', fontsize=9)

    # 添加ATM线
    ax4.axvline(x=1.0, color='blue', linestyle='--', linewidth=2, alpha=0.7)

    # =========================================================================
    # 5. 模型参数展示
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    param_text = f'Model: {model}\n'
    param_text += f'Spot Price: {spot:.2f}\n'
    param_text += f'Surface Size: {len(surface.strikes)} × {len(surface.maturities)}\n\n'

    if model == 'SABR':
        param_text += 'SABR Parameters (by maturity):\n'
        param_text += '-' * 40 + '\n'

        # 显示前8个到期日的参数
        for i, (T, p) in enumerate(list(params.items())[:8]):
            param_text += f'T={T:.2f}y: α={p["alpha"]:.4f}, '
            param_text += f'ρ={p["rho"]:.3f}, ν={p["nu"]:.3f}\n'

        if len(params) > 8:
            param_text += f'... (total {len(params)} maturities)\n'

    else:  # Heston
        param_text += 'Heston Parameters:\n'
        param_text += '-' * 40 + '\n'
        param_text += f'v0 (initial variance):    {params["v0"]:.6f}\n'
        param_text += f'κ (mean reversion):       {params["kappa"]:.4f}\n'
        param_text += f'θ (long-term variance):   {params["theta"]:.6f}\n'
        param_text += f'σ_v (vol of vol):         {params["sigma_v"]:.4f}\n'
        param_text += f'ρ (correlation):          {params["rho"]:.4f}\n'

        if 'calibration_error' in params:
            param_text += f'\nCalibration RMSE: {np.sqrt(params["calibration_error"]):.6f}\n'

    ax5.text(0.05, 0.95, param_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # =========================================================================
    # 6. 统计信息
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    stats_text = 'Surface Statistics:\n'
    stats_text += '-' * 40 + '\n'
    stats_text += f'Min Volatility:  {surface.implied_vols.min():.2%}\n'
    stats_text += f'Max Volatility:  {surface.implied_vols.max():.2%}\n'
    stats_text += f'Mean Volatility: {surface.implied_vols.mean():.2%}\n'
    stats_text += f'Std Volatility:  {surface.implied_vols.std():.2%}\n\n'

    stats_text += 'Strike Range:\n'
    stats_text += f'  Min: {surface.strikes.min():.0f}\n'
    stats_text += f'  Max: {surface.strikes.max():.0f}\n'
    stats_text += f'  Count: {len(surface.strikes)}\n\n'

    stats_text += 'Maturity Range:\n'
    stats_text += f'  Min: {surface.maturities.min():.2f} years\n'
    stats_text += f'  Max: {surface.maturities.max():.2f} years\n'
    stats_text += f'  Count: {len(surface.maturities)}\n\n'

    # ATM波动率（1年期）
    if 0.9 <= 1.0 <= surface.maturities.max():
        atm_vol_1y = surface.get_vol(spot, 1.0)
        stats_text += f'ATM Vol (1Y): {atm_vol_1y:.2%}\n'

    ax6.text(0.05, 0.95, stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 保存和显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_sabr_parameters(result, save_path=None, show=True):
    """
    专门用于SABR模型的参数可视化

    Args:
        result: calibrate_vol_surface() 输出（SABR模型）
        save_path: 保存路径
        show: 是否显示
    """

    if result['model'] != 'SABR':
        print("此函数仅适用于SABR模型")
        return

    params = result['params']

    # 提取参数
    maturities = sorted(params.keys())
    alphas = [params[T]['alpha'] for T in maturities]
    betas = [params[T]['beta'] for T in maturities]
    rhos = [params[T]['rho'] for T in maturities]
    nus = [params[T]['nu'] for T in maturities]

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SABR Parameters by Maturity', fontsize=16, fontweight='bold')

    # Alpha
    axes[0, 0].plot(maturities, alphas, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Maturity (years)')
    axes[0, 0].set_ylabel('Alpha (α)')
    axes[0, 0].set_title('Initial Volatility Level')
    axes[0, 0].grid(True, alpha=0.3)

    # Beta
    axes[0, 1].plot(maturities, betas, 'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('Maturity (years)')
    axes[0, 1].set_ylabel('Beta (β)')
    axes[0, 1].set_title('CEV Parameter')
    axes[0, 1].grid(True, alpha=0.3)

    # Rho
    axes[1, 0].plot(maturities, rhos, 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('Maturity (years)')
    axes[1, 0].set_ylabel('Rho (ρ)')
    axes[1, 0].set_title('Correlation')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # Nu
    axes[1, 1].plot(maturities, nus, 'o-', linewidth=2, markersize=6, color='red')
    axes[1, 1].set_xlabel('Maturity (years)')
    axes[1, 1].set_ylabel('Nu (ν)')
    axes[1, 1].set_title('Vol of Vol')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SABR参数图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# 示例用法
if __name__ == "__main__":
    from yahoo_option_data_cleaner import extract_options_data
    from vol_surface_calibrator import calibrate_vol_surface

    # 加载数据
    df = extract_options_data("spx_infvol_20260109.xlsx")

    # 校准SABR
    print("\nCalibrating SABR...")
    sabr_result = calibrate_vol_surface(df, model='SABR', output_strikes=60)

    # 绘制完整图表
    plot_volatility_surface(sabr_result,
                            save_path='sabr_surface_analysis.png',
                            show=True)

    # 绘制SABR参数
    plot_sabr_parameters(sabr_result,
                        save_path='sabr_parameters.png',
                        show=True)