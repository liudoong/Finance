"""
超级激进的数据过滤 - 修复期限结构跳跃问题

针对您的曲面问题：
1. Term structure 有极端跳跃和V型坑
2. Min IV = 1% (异常)
3. 某些到期日数据质量极差

这个版本会：
- 按到期日分组检查质量
- 移除整个质量差的到期日
- 更严格的 IV 范围（10%-80%）
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def ultra_aggressive_filter(df,
                            min_time_value_pct=0.03,
                            moneyness_range=(0.80, 1.25),
                            min_iv_pct=0.10,  # 10% 最小IV（之前5%）
                            max_iv_pct=0.80,  # 80% 最大IV（之前200%）
                            min_options_per_maturity=10,
                            verbose=True):
    """
    超级激进过滤，专门修复期限结构跳跃问题。

    Args:
        min_time_value_pct: 最小时间价值 3% (更严格)
        moneyness_range: (0.80, 1.25) 更窄，只保留接近ATM
        min_iv_pct: 最小IV 10% (排除异常低IV)
        max_iv_pct: 最大IV 80% (排除异常高IV)
        min_options_per_maturity: 每个到期日最少期权数

    Returns:
        过滤后的 DataFrame
    """

    if verbose:
        print("\n" + "="*80)
        print("超级激进数据过滤 - 修复期限结构问题")
        print("="*80)
        print(f"原始数据: {len(df)} 个期权\n")

    df_clean = df.copy()

    # ===== 步骤 1: 基础清洗 =====
    df_clean = df_clean[
        (df_clean['option_price'] > 0) &
        (df_clean['spot_price'] > 0) &
        (df_clean['strike'] > 0)
    ].copy()

    # 计算内在价值和时间价值
    df_clean['intrinsic'] = df_clean.apply(
        lambda x: max(x['spot_price'] - x['strike'], 0) if x['option_type'].upper() == 'CALL'
        else max(x['strike'] - x['spot_price'], 0),
        axis=1
    )
    df_clean['time_value'] = df_clean['option_price'] - df_clean['intrinsic']
    df_clean['time_value_pct'] = df_clean['time_value'] / df_clean['spot_price']

    # 移除负时间价值
    initial = len(df_clean)
    df_clean = df_clean[df_clean['time_value'] >= 0].copy()
    if verbose and len(df_clean) < initial:
        print(f"[1] 移除负时间价值: {initial - len(df_clean)} 个")

    # ===== 步骤 2: 时间价值过滤（更严格：3%）=====
    initial = len(df_clean)
    df_clean = df_clean[df_clean['time_value_pct'] >= min_time_value_pct].copy()
    if verbose and len(df_clean) < initial:
        print(f"[2] 移除时间价值不足（<{min_time_value_pct*100:.0f}%）: {initial - len(df_clean)} 个")

    # ===== 步骤 3: Moneyness 过滤（更窄：0.80-1.25）=====
    df_clean['moneyness'] = df_clean['strike'] / df_clean['spot_price']
    initial = len(df_clean)
    df_clean = df_clean[
        (df_clean['moneyness'] >= moneyness_range[0]) &
        (df_clean['moneyness'] <= moneyness_range[1])
    ].copy()
    if verbose and len(df_clean) < initial:
        print(f"[3] 移除极端moneyness（保留{moneyness_range[0]}-{moneyness_range[1]}）: {initial - len(df_clean)} 个")

    # ===== 步骤 4: 计算隐含波动率 =====
    if 'implied_vol' not in df_clean.columns:
        if verbose:
            print(f"[4] 计算隐含波动率...")

        from smooth_calibrator import _implied_volatility

        df_clean['implied_vol'] = df_clean.apply(
            lambda row: _implied_volatility(
                row['option_price'],
                row['spot_price'],
                row['strike'],
                row.get('time_to_maturity', 0.5),
                0.05,
                row['option_type'].lower()
            ),
            axis=1
        )

    # 移除IV计算失败的
    initial = len(df_clean)
    df_clean = df_clean[df_clean['implied_vol'].notna()].copy()
    if verbose and len(df_clean) < initial:
        print(f"    移除IV计算失败: {initial - len(df_clean)} 个")

    # ===== 步骤 5: IV 范围过滤（10%-80%，非常严格）=====
    initial = len(df_clean)
    df_clean = df_clean[
        (df_clean['implied_vol'] >= min_iv_pct) &
        (df_clean['implied_vol'] <= max_iv_pct)
    ].copy()
    if verbose and len(df_clean) < initial:
        print(f"[5] 移除IV异常（保留{min_iv_pct*100:.0f}%-{max_iv_pct*100:.0f}%）: {initial - len(df_clean)} 个")

    # ===== 步骤 6: 按到期日检查质量，移除坏的到期日 =====
    if 'maturity_date' in df_clean.columns:
        if verbose:
            print(f"\n[6] 检查每个到期日的数据质量...")

        good_maturities = []
        bad_maturities = []

        for mat, group in df_clean.groupby('maturity_date'):
            # 检查这个到期日的数据质量
            if len(group) < min_options_per_maturity:
                bad_maturities.append(mat)
                continue

            # 检查IV的标准差（太大说明数据不一致）
            iv_std = group['implied_vol'].std()
            iv_mean = group['implied_vol'].mean()

            # 检查IV的范围（太窄说明数据单调）
            iv_range = group['implied_vol'].max() - group['implied_vol'].min()

            # 质量标准
            if iv_std > 0.3:  # 标准差 > 30%，说明数据太分散
                bad_maturities.append(mat)
            elif iv_range < 0.02:  # 范围 < 2%，说明数据太平
                bad_maturities.append(mat)
            elif iv_mean < 0.05 or iv_mean > 1.0:  # 平均IV异常
                bad_maturities.append(mat)
            else:
                good_maturities.append(mat)

        # 保留好的到期日
        initial = len(df_clean)
        df_clean = df_clean[df_clean['maturity_date'].isin(good_maturities)].copy()

        if verbose:
            print(f"    保留到期日: {len(good_maturities)} 个")
            print(f"    移除到期日: {len(bad_maturities)} 个（数据质量差）")
            if len(df_clean) < initial:
                print(f"    移除期权数: {initial - len(df_clean)} 个")

    # ===== 步骤 7: 每个到期日内的IQR离群值移除 =====
    initial = len(df_clean)

    def remove_iqr_outliers(group):
        if len(group) < 5:
            return group

        Q1 = group['implied_vol'].quantile(0.25)
        Q3 = group['implied_vol'].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return group[
            (group['implied_vol'] >= lower) &
            (group['implied_vol'] <= upper)
        ]

    if 'maturity_date' in df_clean.columns:
        df_clean = df_clean.groupby('maturity_date', group_keys=False).apply(
            remove_iqr_outliers
        ).reset_index(drop=True)

        if verbose and len(df_clean) < initial:
            print(f"[7] IQR离群值移除: {initial - len(df_clean)} 个")

    # ===== 最终统计 =====
    if verbose:
        print("\n" + "="*80)
        print(f"✓ 过滤完成")
        print(f"  原始: {len(df)} 个期权")
        print(f"  保留: {len(df_clean)} 个期权 ({len(df_clean)/len(df)*100:.1f}%)")
        print(f"  移除: {len(df) - len(df_clean)} 个期权")

        if len(df_clean) > 0:
            print(f"\n清洗后数据统计:")
            print(f"  Moneyness: {df_clean['moneyness'].min():.3f} - {df_clean['moneyness'].max():.3f}")
            print(f"  时间价值%: {df_clean['time_value_pct'].min()*100:.2f}% - {df_clean['time_value_pct'].max()*100:.2f}%")
            if 'implied_vol' in df_clean.columns:
                print(f"  IV范围: {df_clean['implied_vol'].min()*100:.1f}% - {df_clean['implied_vol'].max()*100:.1f}%")
                print(f"  IV中位数: {df_clean['implied_vol'].median()*100:.1f}%")
                print(f"  IV标准差: {df_clean['implied_vol'].std()*100:.1f}%")

            if 'maturity_date' in df_clean.columns:
                print(f"  到期日数: {df_clean['maturity_date'].nunique()} 个")
                print(f"  平均期权/到期日: {len(df_clean) / df_clean['maturity_date'].nunique():.1f}")

        print("="*80 + "\n")

    # 清理临时列
    df_clean = df_clean.drop(['intrinsic', 'time_value', 'time_value_pct', 'moneyness'],
                              axis=1, errors='ignore')

    return df_clean


if __name__ == "__main__":
    from option_data_cleaner import extract_options_data

    print("加载数据...")
    df = extract_options_data("spx_data.xlsx")

    print(f"\n原始数据IV统计（如果有）:")
    if 'implied_vol' in df.columns:
        print(f"  Min: {df['implied_vol'].min()*100:.1f}%")
        print(f"  Max: {df['implied_vol'].max()*100:.1f}%")
        print(f"  Median: {df['implied_vol'].median()*100:.1f}%")

    # 超级激进过滤
    df_clean = ultra_aggressive_filter(
        df,
        min_time_value_pct=0.03,
        moneyness_range=(0.80, 1.25),
        min_iv_pct=0.10,
        max_iv_pct=0.80,
        min_options_per_maturity=10,
        verbose=True
    )

    # 保存
    df_clean.to_csv('options_ultra_clean.csv', index=False)
    print(f"✓ 已保存: options_ultra_clean.csv")

    # 建议
    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("""
1. 使用这个清洗后的数据重新校准
2. 如果保留率太低（<15%），可以放宽：
   - moneyness_range=(0.75, 1.30)
   - min_iv_pct=0.08
   - max_iv_pct=1.00
3. 检查期限结构是否平滑
    """)