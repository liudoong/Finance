"""
质量优先的期权数据采样器

核心理念：
1. 采样和过滤同时进行
2. 只选择高质量、流动性好的期权
3. 确保每个到期日都有足够的优质数据
4. 在采样阶段就计算和验证 IV

这个模块替代了 data_subsampler.py + smooth_calibrator 的过滤部分
"""

import pandas as pd
import numpy as np
from scipy.stats import norm


def calculate_implied_vol_simple(S, K, T, r, price, option_type='call'):
    """
    简化的隐含波动率计算（Newton-Raphson）
    """
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return np.nan

    # 内在价值检查
    if option_type.lower() == 'call':
        intrinsic = max(S - K * np.exp(-r*T), 0)
    else:
        intrinsic = max(K * np.exp(-r*T) - S, 0)

    if price < intrinsic * 0.99:
        return np.nan

    # 初始猜测
    sigma = 0.3

    for _ in range(50):
        # Black-Scholes 价格
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type.lower() == 'call':
            bs_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            bs_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        diff = price - bs_price

        if abs(diff) < 0.001:
            return sigma

        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T)

        if vega < 1e-10:
            return np.nan

        sigma = sigma + diff / vega
        sigma = max(0.01, min(sigma, 3.0))

    return np.nan


def quality_subsample_options(df,
                              num_strikes_per_maturity=40,
                              min_time_value_pct=0.03,
                              moneyness_range=(0.85, 1.20),
                              min_iv=0.10,
                              max_iv=0.60,
                              min_options_per_maturity=15,
                              risk_free_rate=0.05):
    """
    质量优先的期权采样 - 采样和过滤同时进行

    策略：
    1. 按到期日分组
    2. 对每个到期日：
       a. 过滤掉明显的坏数据
       b. 计算 IV
       c. 根据质量评分选择最好的期权
       d. 确保在 moneyness 维度均匀分布
    3. 移除质量差的整个到期日

    Args:
        num_strikes_per_maturity: 每个到期日目标期权数
        min_time_value_pct: 最小时间价值 3%
        moneyness_range: (0.85, 1.20) 聚焦 ATM
        min_iv: 最小 IV 10%
        max_iv: 最大 IV 60%
        min_options_per_maturity: 每个到期日最少期权数

    Returns:
        高质量的期权 DataFrame
    """

    print("\n" + "="*80)
    print("质量优先采样 - 采样+过滤一体化")
    print("="*80)
    print(f"原始数据: {len(df)} 个期权")
    print(f"目标: 每个到期日 {num_strikes_per_maturity} 个高质量期权\n")

    # 准备数据
    df_work = df.copy()

    # 计算到期时间
    if 'time_to_maturity' not in df_work.columns:
        df_work['trading_date'] = pd.to_datetime(df_work['trading_date'])
        df_work['maturity_date'] = pd.to_datetime(df_work['maturity_date'])
        df_work['days_to_maturity'] = (df_work['maturity_date'] - df_work['trading_date']).dt.days
        df_work['time_to_maturity'] = df_work['days_to_maturity'] / 365.0

    # 基础过滤
    df_work = df_work[
        (df_work['option_price'] > 0) &
        (df_work['spot_price'] > 0) &
        (df_work['strike'] > 0) &
        (df_work['time_to_maturity'] > 0.02) &
        (df_work['time_to_maturity'] < 5.0)
    ].copy()

    print(f"基础验证后: {len(df_work)} 个期权")

    # 按到期日处理
    unique_maturities = sorted(df_work['maturity_date'].unique())
    print(f"到期日数量: {len(unique_maturities)}")

    selected_data = []
    maturity_stats = []

    for mat_date in unique_maturities:
        df_mat = df_work[df_work['maturity_date'] == mat_date].copy()

        if len(df_mat) == 0:
            continue

        # Step 1: 计算内在价值和时间价值
        df_mat['intrinsic'] = df_mat.apply(
            lambda x: max(x['spot_price'] - x['strike'], 0) if x['option_type'].upper() == 'CALL'
            else max(x['strike'] - x['spot_price'], 0),
            axis=1
        )
        df_mat['time_value'] = df_mat['option_price'] - df_mat['intrinsic']
        df_mat['time_value_pct'] = df_mat['time_value'] / df_mat['spot_price']

        # Step 2: 第一轮过滤 - 移除明显坏数据
        df_mat = df_mat[
            (df_mat['time_value'] >= 0) &  # 非负时间价值
            (df_mat['time_value_pct'] >= min_time_value_pct) &  # 足够的时间价值
            (df_mat['option_price'] < df_mat['spot_price'] * 1.5)  # 价格合理
        ].copy()

        if len(df_mat) < 5:
            continue

        # Step 3: Moneyness 过滤
        df_mat['moneyness'] = df_mat['strike'] / df_mat['spot_price']
        df_mat = df_mat[
            (df_mat['moneyness'] >= moneyness_range[0]) &
            (df_mat['moneyness'] <= moneyness_range[1])
        ].copy()

        if len(df_mat) < 5:
            continue

        # Step 4: 计算隐含波动率
        S = df_mat['spot_price'].iloc[0]
        T = df_mat['time_to_maturity'].iloc[0]

        df_mat['implied_vol'] = df_mat.apply(
            lambda row: calculate_implied_vol_simple(
                S, row['strike'], T, risk_free_rate,
                row['option_price'], row['option_type'].lower()
            ),
            axis=1
        )

        # Step 5: IV 过滤
        df_mat = df_mat[
            (df_mat['implied_vol'].notna()) &
            (df_mat['implied_vol'] >= min_iv) &
            (df_mat['implied_vol'] <= max_iv)
        ].copy()

        if len(df_mat) < min_options_per_maturity:
            # 这个到期日质量太差，整个放弃
            continue

        # Step 6: 质量评分
        # 评分标准：
        # - 接近 ATM (moneyness 接近 1.0)
        # - 时间价值充足
        # - IV 在合理范围
        df_mat['atm_distance'] = np.abs(np.log(df_mat['moneyness']))
        df_mat['iv_score'] = 1.0 - np.abs(df_mat['implied_vol'] - 0.20) / 0.40  # 偏好 20% IV
        df_mat['tv_score'] = np.clip(df_mat['time_value_pct'] / 0.05, 0, 1)  # 时间价值越高越好

        # 综合评分
        df_mat['quality_score'] = (
            3.0 / (1.0 + df_mat['atm_distance']) +  # ATM 权重最高
            df_mat['iv_score'] +
            df_mat['tv_score']
        )

        # Step 7: 智能采样 - 在 moneyness 维度均匀分布，但优先高质量
        n_target = min(num_strikes_per_maturity, len(df_mat))

        # 将 moneyness 分成若干个桶
        n_bins = min(n_target // 3, 15)

        try:
            df_mat['moneyness_bin'] = pd.qcut(
                df_mat['moneyness'],
                q=n_bins,
                labels=False,
                duplicates='drop'
            )
        except:
            # 如果分桶失败，直接按质量选择
            selected = df_mat.nlargest(n_target, 'quality_score')
            selected_data.append(selected)

            maturity_stats.append({
                'maturity': mat_date,
                'time_to_maturity': T,
                'n_options': len(selected),
                'iv_mean': selected['implied_vol'].mean(),
                'iv_std': selected['implied_vol'].std(),
                'moneyness_range': (selected['moneyness'].min(), selected['moneyness'].max())
            })
            continue

        # 从每个桶选择最高质量的期权
        selected = []
        for bin_id in df_mat['moneyness_bin'].unique():
            df_bin = df_mat[df_mat['moneyness_bin'] == bin_id]
            # 每个桶选 2-3 个最好的
            n_from_bin = max(1, n_target // n_bins)
            selected.append(df_bin.nlargest(n_from_bin, 'quality_score'))

        selected = pd.concat(selected, ignore_index=True)

        # 如果还不够，补充最高质量的
        if len(selected) < n_target:
            remaining = df_mat[~df_mat.index.isin(selected.index)]
            additional = remaining.nlargest(n_target - len(selected), 'quality_score')
            selected = pd.concat([selected, additional], ignore_index=True)

        # 如果太多，只保留最好的
        if len(selected) > n_target:
            selected = selected.nlargest(n_target, 'quality_score')

        selected_data.append(selected)

        # 统计
        maturity_stats.append({
            'maturity': mat_date,
            'time_to_maturity': T,
            'n_options': len(selected),
            'iv_mean': selected['implied_vol'].mean(),
            'iv_std': selected['implied_vol'].std(),
            'moneyness_range': (selected['moneyness'].min(), selected['moneyness'].max())
        })

    # 合并结果
    if len(selected_data) == 0:
        raise ValueError("没有找到任何高质量数据！数据质量极差。")

    result = pd.concat(selected_data, ignore_index=True)

    # 清理临时列
    columns_to_drop = ['intrinsic', 'time_value', 'time_value_pct', 'moneyness',
                       'atm_distance', 'iv_score', 'tv_score', 'quality_score', 'moneyness_bin']
    result = result.drop(columns_to_drop, axis=1, errors='ignore')

    # 排序
    result = result.sort_values(['maturity_date', 'strike']).reset_index(drop=True)

    # 打印统计
    print("\n" + "="*80)
    print("质量采样完成")
    print("="*80)
    print(f"✓ 选中: {len(result)} 个高质量期权 ({len(result)/len(df)*100:.1f}%)")
    print(f"✓ 保留到期日: {len(maturity_stats)} / {len(unique_maturities)} 个")
    print(f"\n每个到期日的质量统计:")
    print("-" * 80)
    print(f"{'到期时间':<12} {'期权数':<8} {'IV均值':<10} {'IV标准差':<10} {'Moneyness范围'}")
    print("-" * 80)

    for stat in maturity_stats[:10]:  # 显示前10个
        print(f"{stat['time_to_maturity']:>8.2f}年    {stat['n_options']:<8} "
              f"{stat['iv_mean']*100:<10.1f}% {stat['iv_std']*100:<10.2f}% "
              f"{stat['moneyness_range'][0]:.3f} - {stat['moneyness_range'][1]:.3f}")

    if len(maturity_stats) > 10:
        print(f"... (共 {len(maturity_stats)} 个到期日)")

    print("-" * 80)

    # 整体统计
    print(f"\n整体数据质量:")
    print(f"  IV 范围: {result['implied_vol'].min()*100:.1f}% - {result['implied_vol'].max()*100:.1f}%")
    print(f"  IV 中位数: {result['implied_vol'].median()*100:.1f}%")
    print(f"  IV 标准差: {result['implied_vol'].std()*100:.1f}%")
    print(f"  每到期日平均期权数: {len(result)/len(maturity_stats):.1f}")

    # 质量检查
    warnings = []
    if result['implied_vol'].min() < 0.08:
        warnings.append(f"  ⚠ 最小IV ({result['implied_vol'].min()*100:.1f}%) 偏低")
    if result['implied_vol'].max() > 0.70:
        warnings.append(f"  ⚠ 最大IV ({result['implied_vol'].max()*100:.1f}%) 偏高")
    if result['implied_vol'].std() > 0.15:
        warnings.append(f"  ⚠ IV标准差 ({result['implied_vol'].std()*100:.1f}%) 较大，数据可能仍有异常")

    if warnings:
        print(f"\n质量警告:")
        for w in warnings:
            print(w)
    else:
        print(f"\n✓ 数据质量检查通过！")

    print("="*80 + "\n")

    return result


# 测试代码
if __name__ == "__main__":
    from option_data_cleaner import extract_options_data

    print("加载原始数据...")
    df_raw = extract_options_data("spx_data.xlsx")

    print(f"\n原始数据: {len(df_raw)} 个期权")
    print(f"到期日数: {df_raw['maturity_date'].nunique()}")

    # 质量采样
    df_quality = quality_subsample_options(
        df_raw,
        num_strikes_per_maturity=40,
        min_time_value_pct=0.03,
        moneyness_range=(0.85, 1.20),
        min_iv=0.10,
        max_iv=0.60,
        min_options_per_maturity=15,
        risk_free_rate=0.05
    )

    # 保存
    df_quality.to_csv('options_quality_sampled.csv', index=False)
    print(f"✓ 已保存: options_quality_sampled.csv")

    print(f"\n建议:")
    print(f"使用这个高质量数据进行校准，应该能获得平滑的曲面。")