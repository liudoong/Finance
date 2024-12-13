#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:27:51 2024

@author: october
"""

import yfinance as yf
import statsmodels.api as sm

y_x_df = yf.download(['AAPL', '^GSPC'],
                       start = '2020-01-01',
                       end   = '2020-12-31')

y_x_df = y_x_df['Adj Close'].pct_change()

y_x_df.dropna(inplace = True)

y_x_df.rename(columns={"^GSPC":"SP500"} , inplace=True)

x_df = y_x_df[['SP500']]
y_df = y_x_df[['AAPL']]

X_df = sm.add_constant(x_df)
model = sm.OLS(y_df, X_df)
results = model.fit()
print(results.summary())