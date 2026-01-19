"""
直接绘制市场隐含波动率曲面

功能：
- 直接使用 spx_data.xlsx 中的 Implied Volatility 列
- 不需要校准，只是可视化原始市场数据
- 可以看到原始数据的真实情况

这个脚本独立于校准流程，用于：
1. 快速查看市场 IV 数据
2. 诊断数据质量问题
3. 与校准后的曲面对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def load_market_iv_data(excel_file):
    """
    从 Excel 加载市场隐含波动率数据
    """
    print("\n" + "="*80)
    print("加载市场隐含波动率数据")
    print("="*80)

    # 读取数据
    option_df = pd.read_excel(excel_file, sheet_name='option')
    price_df = pd.read_excel(excel_file, sheet_name='price')

    print(f"\n原始数据:")
    print(f"  期权数据: {len(option_df)} 行")
    print(f"  列名: {option_df.columns.tolist()}")

    # 检查是否有 Implied Volatility 列
    iv_col = None
    for col in option_df.columns:
        if 'implied' in col.lower() and 'volatility' in col.lower():
            iv_col = col
            break

    if iv_col is None:
        # 尝试其他可能的列名
        if 'Implied Volatility' in option_df.columns:
            iv_col = 'Implied Volatility'
        else:
            raise ValueError(f"找不到隐含波动率列！现有列: {option_df.columns.tolist()}")

    print(f"  ✓ 找到隐含波动率列: '{iv_col}'")

    # 提取需要的列
    data = {
        'contract': option_df['Contract Name'],
        'last_trade': option_df['Last Trade Date (EST)'],
        'strike': option_df['Strike'],
        'last_price': option_df['Last Price'],
        'implied_vol': option_df[iv_col]
    }

    df = pd.DataFrame(data)

    # 解析合约名称
    contract_parsed = df['contract'].str.extract(
        r'([A-Z]{1,5})W?(\d{6})([CP])(\d{8})'
    )

    df['ticker'] = contract_parsed[0]
    df['option_type'] = contract_parsed[2].map({'C': 'CALL', 'P': 'PUT'})

    # 解析到期日 (YYMMDD)
    maturity_str = contract_parsed[1]
    df['maturity_date'] = pd.to_datetime(
        '20' + maturity_str.str[:2] + maturity_str.str[2:4] + maturity_str.str[4:6],
        format='%Y%m%d',
        errors='coerce'
    )

    # 解析交易日期
    df['trading_date'] = df['last_trade'].str.extract(r'(\d{1,2}/\d{1,2}/\d{4})')[0]
    df['trading_date'] = pd.to_datetime(df['trading_date'], format='%m/%d/%Y', errors='coerce')

    # 获取现货价格（使用最新的收盘价）
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    latest_price = price_df.sort_values('Date', ascending=False).iloc[0]
    spot_price = latest_price['Close\xa0']  # Close 列有不间断空格

    df['spot_price'] = spot_price

    # 计算到期时间
    df['days_to_maturity'] = (df['maturity_date'] - df['trading_date']).dt.days
    df['time_to_maturity'] = df['days_to_maturity'] / 365.0

    # 计算 moneyness
    df['moneyness'] = df['strike'] / df['spot_price']

    # 转换 IV（如果是百分比字符串）
    if df['implied_vol'].dtype == 'object':
        # 可能是 "15.5%" 格式
        df['implied_vol'] = df['implied_vol'].str.replace('%', '').astype(float) / 100.0
    elif df['implied_vol'].max() > 10:
        # 可能是 15.5 (表示15.5%)，需要除以100
        df['implied_vol'] = df['implied_vol'] / 100.0

    # 基础过滤
    df = df[
        (df['implied_vol'].notna()) &
        (df['implied_vol'] > 0) &
        (df['time_to_maturity'] > 0) &
        (df['strike'] > 0)
    ].copy()

    print(f"\n有效数据统计:")
    print(f"  总期权数: {len(df)}")
    print(f"  到期日数: {df['maturity_date'].nunique()}")
    print(f"  IV 范围: {df['implied_vol'].min()*100:.1f}% - {df['implied_vol'].max()*100:.1f}%")
    print(f"  IV 中位数: {df['implied_vol'].median()*100:.1f}%")
    print(f"  Moneyness 范围: {df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}")
    print(f"  到期时间范围: {df['time_to_maturity'].min():.2f} - {df['time_to_maturity'].max():.2f} 年")
    print(f"  现货价格: ${spot_price:.2f}")

    return df


def plot_market_iv_surface(df, title="Market Implied Volatility Surface",
                           save_path='market_iv_surface.png'):
    """
    绘制市场隐含波动率曲面
    """
    print("\n" + "="*80)
    print("绘制市场IV曲面")
    print("="*80)

    spot = df['spot_price'].iloc[0]

    # 创建图表
    fig = plt.figure(figsize=(16, 10))

    # 1. 3D 曲面图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # 使用原始数据点绘制散点图（更真实）
    scatter = ax1.scatter(
        df['strike'],
        df['time_to_maturity'],
        df['implied_vol'],
        c=df['implied_vol'],
        cmap='viridis',
        s=20,
        alpha=0.6
    )

    ax1.set_xlabel('Strike', fontsize=10)
    ax1.set_ylabel('Maturity (years)', fontsize=10)
    ax1.set_zlabel('Implied Volatility', fontsize=10)
    ax1.set_title('3D Market IV (Scatter)', fontsize=12)
    plt.colorbar(scatter, ax=ax1, shrink=0.5)

    # 2. Volatility Smile（不同到期日）
    ax2 = fig.add_subplot(2, 3, 2)

    # 选择几个到期日绘制 smile
    unique_maturities = sorted(df['time_to_maturity'].unique())
    selected_maturities = []

    # 选择大约5个均匀分布的到期日
    if len(unique_maturities) >= 5:
        indices = np.linspace(0, len(unique_maturities)-1, 5, dtype=int)
        selected_maturities = [unique_maturities[i] for i in indices]
    else:
        selected_maturities = unique_maturities

    for mat in selected_maturities:
        df_mat = df[abs(df['time_to_maturity'] - mat) < 0.01]  # 容差
        if len(df_mat) > 3:
            df_mat = df_mat.sort_values('moneyness')
            ax2.plot(df_mat['moneyness'], df_mat['implied_vol'],
                    'o-', label=f'T={mat:.2f}y', alpha=0.7, markersize=4)

    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax2.set_xlabel('Moneyness (K/S)', fontsize=10)
    ax2.set_ylabel('Implied Volatility', fontsize=10)
    ax2.set_title('Volatility Smile (Market Data)', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Term Structure（ATM）
    ax3 = fig.add_subplot(2, 3, 3)

    # 对每个到期日，找接近 ATM 的期权
    atm_data = []
    for mat in unique_maturities:
        df_mat = df[abs(df['time_to_maturity'] - mat) < 0.01]
        if len(df_mat) > 0:
            # 找最接近 ATM 的
            df_mat['atm_dist'] = abs(df_mat['moneyness'] - 1.0)
            closest_atm = df_mat.nsmallest(3, 'atm_dist')  # 取前3个平均
            atm_iv = closest_atm['implied_vol'].mean()
            atm_data.append({'maturity': mat, 'atm_iv': atm_iv})

    if atm_data:
        atm_df = pd.DataFrame(atm_data).sort_values('maturity')
        ax3.plot(atm_df['maturity'], atm_df['atm_iv'], 'o-', color='blue', linewidth=2, markersize=6)
        ax3.set_xlabel('Maturity (years)', fontsize=10)
        ax3.set_ylabel('ATM Implied Volatility', fontsize=10)
        ax3.set_title('ATM Term Structure (Market Data)', fontsize=12)
        ax3.grid(True, alpha=0.3)

    # 4. Heatmap
    ax4 = fig.add_subplot(2, 3, 4)

    # 创建网格数据
    # 按 moneyness 和 maturity 分箱
    moneyness_bins = np.linspace(df['moneyness'].quantile(0.05), df['moneyness'].quantile(0.95), 30)
    maturity_bins = np.linspace(df['time_to_maturity'].min(), df['time_to_maturity'].max(), 20)

    # 创建网格
    grid_iv = np.full((len(maturity_bins)-1, len(moneyness_bins)-1), np.nan)

    for i in range(len(maturity_bins)-1):
        for j in range(len(moneyness_bins)-1):
            mask = (
                (df['time_to_maturity'] >= maturity_bins[i]) &
                (df['time_to_maturity'] < maturity_bins[i+1]) &
                (df['moneyness'] >= moneyness_bins[j]) &
                (df['moneyness'] < moneyness_bins[j+1])
            )
            if mask.sum() > 0:
                grid_iv[i, j] = df.loc[mask, 'implied_vol'].mean()

    im = ax4.imshow(grid_iv, aspect='auto', origin='lower', cmap='viridis',
                    extent=[moneyness_bins[0], moneyness_bins[-1],
                           maturity_bins[0], maturity_bins[-1]])
    ax4.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ATM')
    ax4.set_xlabel('Moneyness (K/S)', fontsize=10)
    ax4.set_ylabel('Maturity (years)', fontsize=10)
    ax4.set_title('IV Heatmap (Market Data)', fontsize=12)
    plt.colorbar(im, ax=ax4, label='Implied Volatility')

    # 5. IV 分布直方图
    ax5 = fig.add_subplot(2, 3, 5)

    ax5.hist(df['implied_vol'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax5.axvline(df['implied_vol'].median(), color='red', linestyle='--',
                linewidth=2, label=f"Median: {df['implied_vol'].median()*100:.1f}%")
    ax5.set_xlabel('Implied Volatility', fontsize=10)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title('IV Distribution', fontsize=12)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. 统计信息
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = f"""
Market IV Surface Statistics

Data Summary:
  Total Options: {len(df)}
  Maturities: {df['maturity_date'].nunique()}
  Spot Price: ${spot:.2f}

Implied Volatility:
  Min: {df['implied_vol'].min()*100:.1f}%
  Max: {df['implied_vol'].max()*100:.1f}%
  Mean: {df['implied_vol'].mean()*100:.1f}%
  Median: {df['implied_vol'].median()*100:.1f}%
  Std Dev: {df['implied_vol'].std()*100:.1f}%

Moneyness Range:
  Min: {df['moneyness'].min():.3f}
  Max: {df['moneyness'].max():.3f}

Maturity Range:
  Min: {df['time_to_maturity'].min():.2f} years
  Max: {df['time_to_maturity'].max():.2f} years

Data Quality Indicators:
  Options with IV > 100%: {(df['implied_vol'] > 1.0).sum()}
  Options with IV < 5%: {(df['implied_vol'] < 0.05).sum()}
  Deep ITM (K/S < 0.7): {(df['moneyness'] < 0.7).sum()}
  Deep OTM (K/S > 1.5): {(df['moneyness'] > 1.5).sum()}
"""

    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 总标题
    fig.suptitle(title + f' (Spot: ${spot:.2f})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {save_path}")

    return fig


def analyze_data_quality(df):
    """
    分析数据质量问题
    """
    print("\n" + "="*80)
    print("数据质量分析")
    print("="*80)

    issues = []

    # 检查异常 IV
    extreme_low = df[df['implied_vol'] < 0.05]
    extreme_high = df[df['implied_vol'] > 1.0]

    if len(extreme_low) > 0:
        issues.append(f"⚠ {len(extreme_low)} 个期权 IV < 5% (异常低)")
    if len(extreme_high) > 0:
        issues.append(f"⚠ {len(extreme_high)} 个期权 IV > 100% (异常高)")

    # 检查 moneyness
    deep_itm = df[df['moneyness'] < 0.7]
    deep_otm = df[df['moneyness'] > 1.5]

    if len(deep_itm) > 0:
        issues.append(f"⚠ {len(deep_itm)} 个期权深度价内 (K/S < 0.7)")
    if len(deep_otm) > 0:
        issues.append(f"⚠ {len(deep_otm)} 个期权深度价外 (K/S > 1.5)")

    # 检查 IV 标准差
    iv_std = df['implied_vol'].std()
    if iv_std > 0.15:
        issues.append(f"⚠ IV 标准差 {iv_std*100:.1f}% 偏大（数据分散）")

    if issues:
        print("\n发现的数据质量问题:")
        for issue in issues:
            print(f"  {issue}")
        print(f"\n建议：")
        print(f"  - 使用 quality_sampler 过滤这些问题数据")
        print(f"  - 限制 moneyness 范围到 0.85-1.20")
        print(f"  - 限制 IV 范围到 10%-60%")
    else:
        print("\n✓ 数据质量良好！")

    return issues


# 主程序
if __name__ == "__main__":
    import sys

    # 输入文件
    input_file = "spx_data.xlsx"

    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    print("="*80)
    print("市场隐含波动率曲面可视化")
    print("="*80)
    print(f"输入文件: {input_file}")

    try:
        # 加载数据
        df = load_market_iv_data(input_file)

        # 分析质量
        issues = analyze_data_quality(df)

        # 绘制曲面
        fig = plot_market_iv_surface(
            df,
            title="SPX Market Implied Volatility Surface",
            save_path='market_iv_surface.png'
        )

        print("\n" + "="*80)
        print("完成！")
        print("="*80)
        print(f"✓ 市场 IV 曲面图: market_iv_surface.png")
        print(f"\n这是原始市场数据的可视化，未经过任何校准或平滑处理。")
        print(f"可以用来:")
        print(f"  1. 快速查看市场数据情况")
        print(f"  2. 识别数据质量问题")
        print(f"  3. 与校准后的曲面对比")

        if issues:
            print(f"\n注意：检测到 {len(issues)} 个数据质量问题")
            print(f"建议使用 clean_workflow.py 进行质量过滤和校准")

        # 显示图表（可选）
        # plt.show()

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()