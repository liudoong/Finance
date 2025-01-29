import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from statsmodels.regression.quantile_regression import QuantReg

def dataset(df, identifier, mpr):
    if identifier == 'ISIN':
        df = df[['Date', 'ISIN', 'PRICE', 'MODDUR_M', 'ZSPRD_M']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d').dt.date
        df.set_index(['Date','ISIN'], inplace = True)
        df['RT'] = df.groupby('ISIN')['PRICE'].pct_change(mpr)
        df = df.dropna()
    elif identifier == 'CUSIP':
        df = df[['Date', 'CUSIP', 'PRICE', 'MODDUR_M', 'ZSPRD_M']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
        df.set_index(['Date', 'CUSIP'], inplace=True)
        df['RT'] = df.groupby('CUSIP')['PRICE'].pct_change(mpr)
        df = df.dropna()
    return df

def tri_cube_kernel(u):
    return np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0)

def gaussian_kernel(u):
    return np.exp(-0.5 * u**2)

def epanechnikov_kernel(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

def local_polynomial_quantile_regression(data, grid, x_vars, y_var, quantile=0.99, bandwidth=1.0, degree=1):
    grid_points = grid[x_vars].values
    data_points = data[x_vars].values
    y = data[y_var].values
    
    distances = np.linalg.norm(data_points[:, np.newaxis, :] - grid_points[np.newaxis, :, :], axis=2)
    
        weights_tri = tri_cube_kernel(distances / bandwidth)
        weights_gauss = gaussian_kernel(distances / bandwidth)
        weights_epan = epanechnikov_kernel(distances / bandwidth)
    
    num_neighborhood_tri = np.sum(weights_tri > 0, axis=0)
    num_neighborhood_gauss = np.sum(weights_gauss > 0, axis=0)
    num_neighborhood_epan = np.sum(weights_epan > 0, axis=0)
    
    X_data = np.column_stack([data_points**d for d in range(degree + 1)])
    X_grid = np.column_stack([grid_points**d for d in range(degree + 1)])
    
    predicted_tri = np.zeros(grid_points.shape[0])
    predicted_gauss = np.zeros(grid_points.shape[0])
    predicted_epan = np.zeros(grid_points.shape[0])
    for i in range(grid_points.shape[0]):
        model_tri = QuantReg(y, X_data)
        result_tri = model_tri.fit(q=quantile, weights=weights_tri[:, i])
        predicted_tri[i] = result_tri.predict(X_grid[i].reshape(1, -1))[0]
        model_gauss = QuantReg(y, X_data)
        result_gauss = model_gauss.fit(q=quantile, weights=weights_gauss[:, i])
        predicted_gauss[i] = result_gauss.predict(X_grid[i].reshape(1, -1))[0]
        
        model_epan = QuantReg(y, X_data)
        result_epan = model_epan.fit(q=quantile, weights=weights_epan[:, i])
        predicted_epan[i] = result_epan.predict(X_grid[i].reshape(1, -1))[0]
    
    output_df = pd.DataFrame({
        'TriCube_Predicted': predicted_tri,
        'TriCube_Neighborhood_Points': num_neighborhood_tri,
        'Gaussian_Predicted': predicted_gauss,
        'Gaussian_Neighborhood_Points': num_neighborhood_gauss,
        'Epanechnikov_Predicted': predicted_epan,
        'Epanechnikov_Neighborhood_Points': num_neighborhood_epan
    })

    return output_df