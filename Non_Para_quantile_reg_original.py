import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import griddata

from statsmodels.regression.quantile_regression import QuantReg

#%%

def dataset(df, identifier, mpr):

if identifier == 'ISIN':

 df = df[['Date', 'ISIN', 'PRICE', 'MODDUR_M', 'ZSPRD_M']].dropna()

 df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d').dt.date

 df.set_index(['Date','ISIN'], inplace = True)

 df['RT'] = df.groupby('ISIN')['PRICE'].pct_change(mpr)

 df = df.dropna()

elif identifier == 'CUSIP':

 df = df[['Date', 'CUSIP', 'PRICE', 'MODDUR_M', 'ZSPRD_M']].dropna()

 df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d').dt.date

 df.set_index(['Date','CUSIP'], inplace = True)

 df['RT'] = df.groupby('CUSIP')['PRICE'].pct_change(mpr)

 df = df.dropna()

return df

def tri_cube_kernel(u):

"""

 Tri-cube kernel function

 """

return np.where(np.abs(u) <=1, (1 - np.abs(u)**3)**3, 0)

def local_polynomial_quantile_regression(data, grid, x_vars, y_var, quantile = 0.99, bandwith = 1.0, degree = 1):

 results = []

for i, row in grid.iterrows():

 x0 = row[x_vars].values

 distances = np.linalg.norm(data[x_vars] - x0, axis=1)

 weights = tri_cube_kernel(distance / bandwidth)

 X = np.column_stack([data[x_vars]**d for d in range(degree +1)])

 y = data[y_var]

 model = QuantReg(y, X)

 result = model.fit(q = quantile, weights = weights)

 x0_poly = np.column_stack([x0**d for d in range(degree + 1)])

 results.append(result.predict(x0_poly.reshape(1, -1))[0])

return results


