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


