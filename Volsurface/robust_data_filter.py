"""
强力数据质量过滤器 - 专门处理真实市场数据的异常值

真实市场数据问题：
1. 深度价内期权：时间价值为负（套利价格）
2. 极端 moneyness：深度价内/价外导致 IV 不稳定
3. 过时报价：Last Price 可能是旧数据
4. 流动性差：Bid-Ask spread 巨大

本模块提供比 smooth_calibrator.py 更严格的过滤。
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


def aggressive_filter_options(df, min_time_value_pct=0.02,
                               moneyness_range=(0.75, 1.35),
                               max_iv_pct=2.0,
                               min_iv_pct=0.05,
                               verbose=True):
    """
    激进的期权数据过滤，专门处理真实市场数据。

    Args:
        df: 期权数据 DataFrame
        min_time_value_pct: 最小时间价值占比 (默认2%，比之前5%更宽松一点)
        moneyness_range: Moneyness 范围 (默认0.75-1.35，聚焦ATM附近)
        max_iv_pct: 最大隐含波动率 (默认200%，过滤异常高IV)
        min_iv_pct: 最小隐含波动率 (默认5%，过滤异常低IV)
        verbose: 是否打印详细信息

    Returns:
        过滤后的 DataFrame
    """
    if verbose:
        print("\n" + "="*80)
        print("激进数据过滤 - 处理真实市场数据异常")
        print("="*80)
        print(f"原始数据: {len(df)} 个期权\n")

    df_clean = df.copy()
    initial_count = len(df_clean)

    # ===== 第1步：基础验证 =====
    df_clean = df_clean[
        (df_clean['option_price'] > 0) &
        (df_clean['spot_price'] > 0) &
        (df_clean['strike'] > 0) &
        (df_clean['option_price'].notna()) &
        (df_clean['spot_price'].notna())
    ].copy()

    if verbose and len(df_clean) < initial_count:
        print(f"[1] 基础验证: 移除 {initial_count - len(df_clean)} 个无效数据")
        print(f"    剩余: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)\n")

    initial_count = len(df_clean)

    # ===== 第2步：计算内在价值和时间价值 =====
    df_clean['intrinsic'] = df_clean.apply(
        lambda x: max(x['spot_price'] - x['strike'], 0) if x['option_type'].upper() == 'CALL'
        else max(x['strike'] - x['spot_price'], 0),
        axis=1
    )

    df_clean['time_value'] = df_clean['option_price'] - df_clean['intrinsic']
    df_clean['time_value_pct'] = df_clean['time_value'] / df_clean['spot_price']

    # 移除时间价值为负的期权（违反无套利）
    df_clean = df_clean[df_clean['time_value'] >= 0].copy()

    if verbose and len(df_clean) < initial_count:
        removed = initial_count - len(df_clean)
        print(f"[2] 移除负时间价值期权（套利价格）: {removed} 个")
        print(f"    剩余: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)\n")

    initial_count = len(df_clean)

    # ===== 第3步：时间价值过滤（更宽松）=====
    df_clean = df_clean[df_clean['time_value_pct'] >= min_time_value_pct].copy()

    if verbose and len(df_clean) < initial_count:
        removed = initial_count - len(df_clean)
        print(f"[3] 移除时间价值不足期权 (<{min_time_value_pct*100:.1f}%): {removed} 个")
        print(f"    剩余: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)\n")

    initial_count = len(df_clean)

    # ===== 第4步：Moneyness 过滤（聚焦ATM）=====
    df_clean['moneyness'] = df_clean['strike'] / df_clean['spot_price']
    df_clean = df_clean[
        (df_clean['moneyness'] >= moneyness_range[0]) &
        (df_clean['moneyness'] <= moneyness_range[1])
    ].copy()

    if verbose and len(df_clean) < initial_count:
        removed = initial_count - len(df_clean)
        print(f"[4] 移除极端moneyness期权 (保留{moneyness_range[0]}-{moneyness_range[1]}): {removed} 个")
        print(f"    剩余: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)\n")

    initial_count = len(df_clean)

    # ===== 第5步：计算隐含波动率并过滤异常 =====
    if 'implied_vol' not in df_clean.columns:
        if verbose:
            print("[5] 计算隐含波动率...")

        from smooth_calibrator import _implied_volatility

        df_clean['implied_vol'] = df_clean.apply(
            lambda row: _implied_volatility(
                row['option_price'],
                row['spot_price'],
                row['strike'],
                row.get('time_to_maturity', 0.25),  # 默认3个月
                0.05,  # 默认利率
                row['option_type'].lower()
            ),
            axis=1
        )

    # 移除 IV 计算失败的
    df_clean = df_clean[df_clean['implied_vol'].notna()].copy()

    # 移除 IV 异常的
    df_clean = df_clean[
        (df_clean['implied_vol'] >= min_iv_pct) &
        (df_clean['implied_vol'] <= max_iv_pct)
    ].copy()

    if verbose and len(df_clean) < initial_count:
        removed = initial_count - len(df_clean)
        print(f"[5] 移除IV异常期权 (保留{min_iv_pct*100:.0f}%-{max_iv_pct*100:.0f}%): {removed} 个")
        print(f"    剩余: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)\n")

    initial_count = len(df_clean)

    # ===== 第6步：按maturity分组，移除每组内的离群值 =====
    def remove_outliers_per_group(group):
        """每个到期日内移除离群值"""
        if len(group) < 5:
            return group

        # 使用 IQR 方法（比 Z-score 更稳健）
        Q1 = group['implied_vol'].quantile(0.25)
        Q3 = group['implied_vol'].quantile(0.75)
        IQR = Q3 - Q1

        # 移除超出 1.5*IQR 范围的
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return group[
            (group['implied_vol'] >= lower_bound) &
            (group['implied_vol'] <= upper_bound)
        ]

    if 'maturity_date' in df_clean.columns:
        df_clean = df_clean.groupby('maturity_date', group_keys=False).apply(
            remove_outliers_per_group
        ).reset_index(drop=True)

        if verbose and len(df_clean) < initial_count:
            removed = initial_count - len(df_clean)
            print(f"[6] 移除每个到期日内的IQR离群值: {removed} 个")
            print(f"    剩余: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)\n")

    initial_count = len(df_clean)

    # ===== 第7步：移除价格不合理的期权 =====
    df_clean = df_clean[
        df_clean['option_price'] < df_clean['spot_price'] * 1.5  # 期权价格不应超过标的1.5倍
    ].copy()

    if verbose and len(df_clean) < initial_count:
        removed = initial_count - len(df_clean)
        print(f"[7] 移除价格不合理期权: {removed} 个")
        print(f"    剩余: {len(df_clean)} ({len(df_clean)/initial_count*100:.1f}%)\n")

    # ===== 最终统计 =====
    if verbose:
        print("="*80)
        print(f"✓ 过滤完成")
        print(f"  原始: {len(df)} 个期权")
        print(f"  保留: {len(df_clean)} 个期权 ({len(df_clean)/len(df)*100:.1f}%)")
        print(f"  移除: {len(df) - len(df_clean)} 个期权\n")

        if len(df_clean) > 0:
            print(f"清洗后数据质量:")
            print(f"  Moneyness 范围: {df_clean['moneyness'].min():.3f} - {df_clean['moneyness'].max():.3f}")
            print(f"  时间价值%: {df_clean['time_value_pct'].min()*100:.2f}% - {df_clean['time_value_pct'].max()*100:.2f}%")
            if 'implied_vol' in df_clean.columns:
                print(f"  隐含波动率: {df_clean['implied_vol'].min()*100:.1f}% - {df_clean['implied_vol'].max()*100:.1f}%")
                print(f"  IV 中位数: {df_clean['implied_vol'].median()*100:.1f}%")
        print("="*80 + "\n")

    # 清理临时列
    df_clean = df_clean.drop(['intrinsic', 'time_value', 'time_value_pct', 'moneyness'],
                              axis=1, errors='ignore')

    return df_clean


def diagnose_data_quality(df):
    """
    诊断数据质量，打印详细报告。
    """
    print("\n" + "="*80)
    print("数据质量诊断报告")
    print("="*80 + "\n")

    # 基础统计
    print(f"1. 基础统计:")
    print(f"   总期权数: {len(df)}")
    print(f"   Call期权: {(df['option_type'].str.upper() == 'CALL').sum()}")
    print(f"   Put期权: {(df['option_type'].str.upper() == 'PUT').sum()}\n")

    # 价格分布
    print(f"2. 期权价格分布:")
    print(f"   平均: ${df['option_price'].mean():.2f}")
    print(f"   中位数: ${df['option_price'].median():.2f}")
    print(f"   最大: ${df['option_price'].max():.2f}")
    print(f"   最小: ${df['option_price'].min():.2f}")
    print(f"   >$1000: {(df['option_price'] > 1000).sum()} 个")
    print(f"   <$1: {(df['option_price'] < 1).sum()} 个\n")

    # Moneyness
    df['moneyness'] = df['strike'] / df['spot_price']
    print(f"3. Moneyness 分布:")
    print(f"   平均: {df['moneyness'].mean():.3f}")
    print(f"   中位数: {df['moneyness'].median():.3f}")
    print(f"   范围: {df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}")
    print(f"   <0.7 (深度ITM): {(df['moneyness'] < 0.7).sum()} 个")
    print(f"   >1.5 (深度OTM): {(df['moneyness'] > 1.5).sum()} 个\n")

    # 时间价值
    df['intrinsic'] = df.apply(
        lambda x: max(x['spot_price'] - x['strike'], 0) if x['option_type'].upper() == 'CALL'
        else max(x['strike'] - x['spot_price'], 0),
        axis=1
    )
    df['time_value'] = df['option_price'] - df['intrinsic']

    print(f"4. 时间价值:")
    print(f"   平均: ${df['time_value'].mean():.2f}")
    print(f"   中位数: ${df['time_value'].median():.2f}")
    print(f"   负值: {(df['time_value'] < 0).sum()} 个 ⚠️")
    print(f"   <$1: {(df['time_value'] < 1).sum()} 个\n")

    # 问题数据
    problems = []
    if (df['time_value'] < 0).sum() > 0:
        problems.append(f"   ⚠️ {(df['time_value'] < 0).sum()} 个期权时间价值为负（违反无套利）")
    if (df['moneyness'] < 0.5).sum() > 0:
        problems.append(f"   ⚠️ {(df['moneyness'] < 0.5).sum()} 个期权深度价内（IV不稳定）")
    if (df['moneyness'] > 2.0).sum() > 0:
        problems.append(f"   ⚠️ {(df['moneyness'] > 2.0).sum()} 个期权深度价外（IV不稳定）")
    if (df['option_price'] > df['spot_price']).sum() > 0:
        problems.append(f"   ⚠️ {(df['option_price'] > df['spot_price']).sum()} 个期权价格超过标的价格")

    if problems:
        print(f"5. 检测到的问题:")
        for p in problems:
            print(p)
    else:
        print(f"5. ✓ 未检测到明显问题")

    print("\n" + "="*80 + "\n")


# 测试代码
if __name__ == "__main__":
    from option_data_cleaner import extract_options_data

    print("加载数据...")
    df = extract_options_data("spx_data.xlsx")

    # 诊断
    diagnose_data_quality(df)

    # 激进过滤
    df_clean = aggressive_filter_options(
        df,
        min_time_value_pct=0.02,
        moneyness_range=(0.75, 1.35),
        max_iv_pct=2.0,
        min_iv_pct=0.05,
        verbose=True
    )

    # 保存
    df_clean.to_csv('options_aggressively_filtered.csv', index=False)
    print(f"已保存清洗后的数据: options_aggressively_filtered.csv")