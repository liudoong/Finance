"""
测试数据过滤效果

对比：
1. 无过滤
2. 标准过滤（smooth_calibrator原版）
3. 激进过滤（robust_data_filter）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from option_data_cleaner import extract_options_data
from robust_data_filter import aggressive_filter_options, diagnose_data_quality

# 加载数据
print("="*80)
print("加载原始数据...")
print("="*80)
df_raw = extract_options_data("spx_data.xlsx")

# 诊断
diagnose_data_quality(df_raw)

# 应用激进过滤
df_clean = aggressive_filter_options(
    df_raw,
    min_time_value_pct=0.02,
    moneyness_range=(0.75, 1.35),
    max_iv_pct=2.0,
    min_iv_pct=0.05,
    verbose=True
)

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Moneyness 分布
ax1 = axes[0, 0]
df_raw['moneyness'] = df_raw['strike'] / df_raw['spot_price']
if 'moneyness' not in df_clean.columns:
    df_clean['moneyness'] = df_clean['strike'] / df_clean['spot_price']

ax1.hist(df_raw['moneyness'], bins=50, alpha=0.5, label='原始数据', color='red')
ax1.hist(df_clean['moneyness'], bins=50, alpha=0.7, label='过滤后', color='green')
ax1.axvline(x=0.75, color='black', linestyle='--', alpha=0.5, label='过滤边界')
ax1.axvline(x=1.35, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('Moneyness (K/S)')
ax1.set_ylabel('数量')
ax1.set_title('Moneyness 分布对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 期权价格分布（log scale）
ax2 = axes[0, 1]
ax2.hist(np.log10(df_raw['option_price'] + 0.01), bins=50, alpha=0.5, label='原始数据', color='red')
ax2.hist(np.log10(df_clean['option_price'] + 0.01), bins=50, alpha=0.7, label='过滤后', color='green')
ax2.set_xlabel('Log10(期权价格)')
ax2.set_ylabel('数量')
ax2.set_title('期权价格分布对比（对数）')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 隐含波动率分布（如果有）
ax3 = axes[1, 0]
if 'implied_vol' in df_clean.columns:
    # 过滤原始数据中的NaN
    iv_raw = df_raw.get('implied_vol', pd.Series([]))
    if len(iv_raw) > 0:
        iv_raw_clean = iv_raw[iv_raw.notna() & (iv_raw > 0) & (iv_raw < 5)]
        ax3.hist(iv_raw_clean, bins=50, alpha=0.5, label='原始数据', color='red', range=(0, 2))

    iv_clean = df_clean['implied_vol']
    ax3.hist(iv_clean, bins=50, alpha=0.7, label='过滤后', color='green', range=(0, 2))
    ax3.axvline(x=0.05, color='black', linestyle='--', alpha=0.5, label='IV边界')
    ax3.axvline(x=2.0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('隐含波动率')
    ax3.set_ylabel('数量')
    ax3.set_title('隐含波动率分布对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'IV 数据未计算', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('隐含波动率分布')

# 4. 统计摘要
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
数据过滤效果总结

原始数据: {len(df_raw)} 个期权
过滤后: {len(df_clean)} 个期权
保留率: {len(df_clean)/len(df_raw)*100:.1f}%
移除: {len(df_raw) - len(df_clean)} 个期权

关键改进:
• 移除负时间价值期权
• 限制moneyness范围 (0.75-1.35)
• 移除极端IV (< 5% 或 > 200%)
• IQR离群值检测

原始数据问题:
• 负时间价值: {(df_raw['option_price'] - np.where(df_raw['option_type'].str.upper() == 'CALL', np.maximum(df_raw['spot_price'] - df_raw['strike'], 0), np.maximum(df_raw['strike'] - df_raw['spot_price'], 0)) < 0).sum()} 个
• 深度价内(<0.7): {(df_raw['moneyness'] < 0.7).sum()} 个
• 深度价外(>1.5): {(df_raw['moneyness'] > 1.5).sum()} 个

过滤后数据质量:
• Moneyness: {df_clean['moneyness'].min():.3f} - {df_clean['moneyness'].max():.3f}
"""

if 'implied_vol' in df_clean.columns:
    summary_text += f"• IV: {df_clean['implied_vol'].min()*100:.1f}% - {df_clean['implied_vol'].max()*100:.1f}%\n"
    summary_text += f"• IV中位数: {df_clean['implied_vol'].median()*100:.1f}%"

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('data_filtering_comparison.png', dpi=150)
print("\n✓ 对比图已保存: data_filtering_comparison.png")
plt.show()

# 保存清洗后的数据
df_clean.to_csv('options_aggressively_filtered.csv', index=False)
print(f"✓ 清洗后数据已保存: options_aggressively_filtered.csv")

print("\n" + "="*80)
print("建议:")
print("="*80)
print(f"""
1. 原始数据有 {len(df_raw) - len(df_clean)} 个问题期权被移除
2. 保留率 {len(df_clean)/len(df_raw)*100:.1f}% - {'较低，数据质量差' if len(df_clean)/len(df_raw) < 0.3 else '正常' if len(df_clean)/len(df_raw) < 0.7 else '高，数据质量好'}
3. 使用清洗后的数据进行校准应该能避免曲面翻转问题

下一步:
• 用 options_aggressively_filtered.csv 重新运行校准
• 或者在 fast_calibration_workflow.py 中设置 FILTER_DATA=True
• 检查生成的曲面图是否平滑单调
""")
print("="*80)