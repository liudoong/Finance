#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:39:55 2023

@author: DLIU
"""


import pandas as pd
import yfinance as yf

#%% Yahoo Finance

df = yf.download("AAPL",
                 start = "2011-01-01",
                 end   = "2021-12-31",
                 progress = False)

print(f"Downloaded {len(df)} rows of data.")
df

df1 = yf.download(["AAPL","MSFT"],
                 start = "2011-01-01",
                 end   = "2021-12-31",
                 progress = True)

df1

df3 = yf.download(["AAPL","MSFT"],
                 start = "2011-01-01",
                 end   = "2021-12-31",
                 progress = False,
                 actions="inline")

df3

aapl_data = yf.Ticker("AAPL")
aapl_data.history()


#%% Nasdaq data

#before downloading the data, we need to create an account at Nasdaq data link
# https://data.naddaq.com
#then authenticate email address
# then find personal API key in profile at http://data.nasdaq.com/account/profile
# xLy_Q1zJtW2h6-E_egAf

import pandas as pd
import nasdaqdatalink

nasdaqdatalink.ApiConfig.api_key = "xLy_Q1zJtW2h6-E_egAf"

df4 = nasdaqdatalink.get(dataset="WIKI/AAPL",
                        start_date = "2011-01-01",
                        end_date   = "2021-12-31")

print(f"Downloaded {len(df3)} rows of data.")
df4.head()

#download multiple tickers using the get_table function

COLUNS = ["ticker", "date", "adj_close"]

df5 = nasdaqdatalink.get_table("WIKI/PRICES")

#%%

import pandas as pd
import pandas_datareader.data as web
import datetime

start_date = datetime.datetime(2000,1,1)
end_date   = datetime.datetime(2022,12,31)

symbol = "GS10"

data = web.DataReader(symbol, "fred", start_date, end_date)




Reading B: PFE × PD^¼
This stays linear in PFE, so the result is still dollars. It’s a PD-weighted exposure that you can sum and compare.
It is concave in PD, so riskier names get more weight, but much less than proportionally:
PD	PD^¼	Weighted PFE on $100m
0.1%	0.178	$17.8m
1%	0.316	$31.6m
10%	0.562	$56.2m

Under this reading, a 100× increase in PD only raises the weight about 3×.

Why this could make sense: it resembles how regulatory capital scales. Under the Basel IRB formula, the capital charge per dollar of EAD rises with PD, but much more slowly than linearly. Plain PD × PFE (expected-loss style) treats a 10% PD name as 100× riskier than a 0.1% PD name, which overstates things from a capital or stress perspective. A fractional power of PD is a crude stand-in for that concave curve. A fourth root is flatter than IRB is over most of the PD range; IRB behaves more like a power somewhere between ¼ and ½. So I’d see Reading B as a simple capital-like proxy, not a recognised model.

What I’d confirm with Jack
Which bracket? Reading B is the only one I’d defend, because it keeps the result in dollars.
What is it for? Ranking counterparties, picking names for stress scenarios, or allocating limits? Choosing the PD exponent only makes sense once the purpose is clear.
Where does “one-year” fit? Neither reading turns a PFE into a one-year PFE. More likely the PD is a one-year PD and the PFE is a one-year-horizon peak, so the product is a one-year weighted exposure. Make sure both inputs actually use the same horizon.
