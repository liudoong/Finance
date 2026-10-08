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





Repo a TBA: no. A repo needs a security you own and can deliver as collateral. A TBA is only a promise to deliver pools on a future settlement date, so before that date there is nothing to pledge. The substitute is the dollar roll (sell the front month, buy back the next month), which gives you the same financing economics. If you want an actual repo, you take delivery of the pools and repo those.

Cash-trade a TBA: yes, in one sense and no in another.

Yes, outright buying and selling. TBAs are the main outright trading market for agency MBS. Street desks quote them, and people trade them directly for exposure, hedging, or relative value. Market participants call this the “cash MBS” market to distinguish it from derivatives like swaps or futures. In that sense TBA trading is cash trading.
No, spot settlement. You can’t settle a TBA T+1 or T+2 like a Treasury. TBAs settle only on the monthly SIFMA settlement dates, one for each product class (Fannie/Freddie 30-year, 15-year, Ginnie, and so on). Pool details are announced 48 hours before settlement. The closest you can get to spot is trading the front month just before its settlement date.

If you want true spot settlement, trade specified pools. A specified pool is an identified pool with a known CUSIP, loan characteristics, and prepayment profile. It trades outright with normal short settlement, usually at a payup over TBA for better prepayment characteristics. It is also what you would put into repo.


