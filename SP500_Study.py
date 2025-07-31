#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 23:00:11 2025

@author: october
"""

from yahooquery import Ticker
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# 1. 下载数据
ticker = Ticker('^GSPC')
data = ticker.history(start='2020-01-01', end=None, interval='1d')

# 2. 整理数据
sp500 = data['close'].reset_index()
sp500['date'] = pd.to_datetime(sp500['date'])
sp500.set_index('date', inplace=True)

# 3. 只保留 Close 列
sp500 = sp500[['close']].copy()

# 4. 找到所有历史新高
sp500['cummax'] = sp500['close'].cummax()
sp500['is_new_high'] = sp500['close'] == sp500['cummax']
new_highs = sp500[sp500['is_new_high']]

print(f"共找到 {len(new_highs)} 个历史新高点")

# 5. 绘图
plt.figure(figsize=(14, 7))
plt.plot(sp500.index, sp500['close'], label='S&P 500 Close Price', color='blue', linewidth=1)

# 标注所有历史新高
plt.scatter(new_highs.index, new_highs['close'], color='red', label='Historical Highs', zorder=5, s=30)

# 图形美化
plt.title('S&P 500 Historical Highs (from 2020-01-01)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Index Level', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%%

from yahooquery import Ticker
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web

# 1. 下载 S&P 500 数据
ticker = Ticker('^GSPC')
data = ticker.history(start='2020-01-01', end=None, interval='1d')
sp500 = data['close'].reset_index()
sp500['date'] = pd.to_datetime(sp500['date'])
sp500.set_index('date', inplace=True)
sp500 = sp500[['close']].copy()

# 2. 下载 M2 和 CPI 数据（FRED）
start_date = sp500.index.min()
end_date = sp500.index.max()
m2 = web.DataReader('M2SL', 'fred', start_date, end_date)
cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# 3. 对 M2 和 CPI 进行标准化（Min-Max）
m2['M2_scaled'] = (m2['M2SL'] - m2['M2SL'].min()) / (m2['M2SL'].max() - m2['M2SL'].min())
cpi['CPI_scaled'] = (cpi['CPIAUCSL'] - cpi['CPIAUCSL'].min()) / (cpi['CPIAUCSL'].max() - cpi['CPIAUCSL'].min())

# 4. 合并所有数据
combined = sp500.join(m2, how='inner').join(cpi, how='inner')

# 5. 绘图
fig, ax1 = plt.subplots(figsize=(16, 9))

# S&P 500
ax1.plot(combined.index, combined['close'], color='blue', label='S&P 500', linewidth=1.8)
ax1.set_ylabel('S&P 500 Index', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# M2 & CPI 副轴
ax2 = ax1.twinx()
ax2.plot(combined.index, combined['M2_scaled'], color='green', linestyle='-', label='M2 Scaled', linewidth=1.5)
ax2.plot(combined.index, combined['CPI_scaled'], color='red', linestyle='--', label='CPI Scaled', linewidth=1.5)
ax2.set_ylabel('M2 & CPI Scaled (0-1)', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# 图例 & 美化
fig.suptitle('S&P 500 vs M2 Money Supply vs CPI Inflation (Standardized)', fontsize=18)
ax1.legend(loc='upper left')
ax2.legend(loc='lower right')
ax1.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

