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
    """
    Preprocess the dataset based on the identifier (ISIN or CUSIP) and calculate the return (RT).

    Parameters:
        df (pd.DataFrame): The input dataset.
        identifier (str): Either 'ISIN' or 'CUSIP' to identify the type of dataset.
        mpr (int): The number of periods for calculating percentage change (return).

    Returns:
        pd.DataFrame: The preprocessed dataset with Date and identifier as the index, and RT (return) calculated.
    """
    if identifier == 'ISIN':
        # Select relevant columns and drop rows with missing values
        df = df[['Date', 'ISIN', 'PRICE', 'MODDUR_M', 'ZSPRD_M']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d').dt.date
        df.set_index(['Date','ISIN'], inplace = True)
        # Convert Date column to datetime format and extract the date part
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
        # Set Date and ISIN as the index
        df.set_index(['Date', 'ISIN'], inplace=True)
        # Calculate the percentage change in PRICE (return) grouped by ISIN
        df['RT'] = df.groupby('ISIN')['PRICE'].pct_change(mpr)
        # Drop rows with missing values after calculating RT
        df = df.dropna()
    elif identifier == 'CUSIP':
        # Select relevant columns and drop rows with missing values
        df = df[['Date', 'CUSIP', 'PRICE', 'MODDUR_M', 'ZSPRD_M']].dropna()
        # Convert Date column to datetime format and extract the date part
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
        # Set Date and CUSIP as the index
        df.set_index(['Date', 'CUSIP'], inplace=True)
        # Calculate the percentage change in PRICE (return) grouped by CUSIP
        df['RT'] = df.groupby('CUSIP')['PRICE'].pct_change(mpr)
        # Drop rows with missing values after calculating RT
        df = df.dropna()
    return df


def tri_cube_kernel(u):
    """
    Tri-cube kernel function.

    Parameters:
        u (np.array): Input array of distances normalized by bandwidth.

    Returns:
        np.array: Weights computed using the tri-cube kernel.
    """
    return np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0)


def gaussian_kernel(u):
    """
    Gaussian kernel function.

    Parameters:
        u (np.array): Input array of distances normalized by bandwidth.

    Returns:
        np.array: Weights computed using the Gaussian kernel.
    """
    return np.exp(-0.5 * u**2)


def epanechnikov_kernel(u):
    """
    Epanechnikov kernel function.

    Parameters:
        u (np.array): Input array of distances normalized by bandwidth.

    Returns:
        np.array: Weights computed using the Epanechnikov kernel.
    """
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)


def local_polynomial_quantile_regression(data, grid, x_vars, y_var, quantile=0.99, bandwidth=1.0, degree=1):
    grid_points = grid[x_vars].values
    data_points = data[x_vars].values
    y = data[y_var].values
    """
    Perform local polynomial quantile regression using multiple kernel functions.


    Parameters:
        data (pd.DataFrame): The input dataset containing the independent and dependent variables.
        grid (pd.DataFrame): The grid points at which predictions are to be made.
        x_vars (list): List of independent variable names.
        y_var (str): Name of the dependent variable.
        quantile (float): The quantile to fit (default is 0.99).
        bandwidth (float): The bandwidth for the kernel functions (default is 1.0).
        degree (int): The degree of the polynomial (default is 1).


    Returns:
        pd.DataFrame: A DataFrame containing predicted results and neighborhood points used for each kernel.
    """

    # Extract grid points and data points as numpy arrays for faster computation
    grid_points = grid[x_vars].values  # Shape: (n_grid_points, n_x_vars)
    data_points = data[x_vars].values  # Shape: (n_data_points, n_x_vars)
    y = data[y_var].values             # Shape: (n_data_points,)

    # Compute distances between all data points and grid points using broadcasting
    # Shape of distances: (n_data_points, n_grid_points)
    distances = np.linalg.norm(data_points[:, np.newaxis, :] - grid_points[np.newaxis, :, :], axis=2)
    
        weights_tri = tri_cube_kernel(distances / bandwidth)
        weights_gauss = gaussian_kernel(distances / bandwidth)
        weights_epan = epanechnikov_kernel(distances / bandwidth)
    # Compute weights for each kernel function
    # Tri-cube kernel weights
    weights_tri = tri_cube_kernel(distances / bandwidth)  # Shape: (n_data_points, n_grid_points)
    # Gaussian kernel weights
    weights_gauss = gaussian_kernel(distances / bandwidth)  # Shape: (n_data_points, n_grid_points)
    # Epanechnikov kernel weights
    weights_epan = epanechnikov_kernel(distances / bandwidth)  # Shape: (n_data_points, n_grid_points)

    num_neighborhood_tri = np.sum(weights_tri > 0, axis=0)
    num_neighborhood_gauss = np.sum(weights_gauss > 0, axis=0)
    num_neighborhood_epan = np.sum(weights_epan > 0, axis=0)
    # Compute the number of neighborhood points for each kernel
    # Count the number of non-zero weights for each grid point
    num_neighborhood_tri = np.sum(weights_tri > 0, axis=0)  # Shape: (n_grid_points,)
    num_neighborhood_gauss = np.sum(weights_gauss > 0, axis=0)  # Shape: (n_grid_points,)
    num_neighborhood_epan = np.sum(weights_epan > 0, axis=0)  # Shape: (n_grid_points,)

    # Construct the design matrix for data points
    # X_data is a matrix where each column is a polynomial term of the independent variables
    # Shape: (n_data_points, degree + 1)
    X_data = np.column_stack([data_points**d for d in range(degree + 1)])

    # Construct the design matrix for grid points
    # X_grid is a matrix where each column is a polynomial term of the grid points
    # Shape: (n_grid_points, degree + 1)
    X_grid = np.column_stack([grid_points**d for d in range(degree + 1)])
    
    predicted_tri = np.zeros(grid_points.shape[0])
    predicted_gauss = np.zeros(grid_points.shape[0])
    predicted_epan = np.zeros(grid_points.shape[0])
    # Initialize arrays to store predictions for each kernel
    predicted_tri = np.zeros(grid_points.shape[0])  # Shape: (n_grid_points,)
    predicted_gauss = np.zeros(grid_points.shape[0])  # Shape: (n_grid_points,)
    predicted_epan = np.zeros(grid_points.shape[0])  # Shape: (n_grid_points,)

    # Fit models and make predictions for each grid point
    for i in range(grid_points.shape[0]):
        # Tri-cube kernel
        model_tri = QuantReg(y, X_data)
        result_tri = model_tri.fit(q=quantile, weights=weights_tri[:, i])
        predicted_tri[i] = result_tri.predict(X_grid[i].reshape(1, -1))[0]
        
        # Gaussian kernel
        model_gauss = QuantReg(y, X_data)
        result_gauss = model_gauss.fit(q=quantile, weights=weights_gauss[:, i])
        predicted_gauss[i] = result_gauss.predict(X_grid[i].reshape(1, -1))[0]
        
        weights_tri = tri_cube_kernel(distances / bandwidth)
        weights_gauss = gaussian_kernel(distances / bandwidth)
        weights_epan = epanechnikov_kernel(distances / bandwidth)
        # Epanechnikov kernel
        model_epan = QuantReg(y, X_data)
        result_epan = model_epan.fit(q=quantile, weights=weights_epan[:, i])
        predicted_epan[i] = result_epan.predict(X_grid[i].reshape(1, -1))[0]
    
    # Create output DataFrame to store results
    output_df = pd.DataFrame({
        'TriCube_Predicted': predicted_tri,
        'TriCube_Neighborhood_Points': num_neighborhood_tri,
        'Gaussian_Predicted': predicted_gauss,
        'Gaussian_Neighborhood_Points': num_neighborhood_gauss,
        'Epanechnikov_Predicted': predicted_epan,
        'Epanechnikov_Neighborhood_Points': num_neighborhood_epan
    num_neighborhood_tri = np.sum(weights_tri > 0, axis=0)
    num_neighborhood_gauss = np.sum(weights_gauss > 0, axis=0)
    num_neighborhood_epan = np.sum(weights_epan > 0, axis=0)

    predicted_tri = np.zeros(grid_points.shape[0])
    predicted_gauss = np.zeros(grid_points.shape[0])
    predicted_epan = np.zeros(grid_points.shape[0])
    })

    })
    return output_df
    return output_df
    return output_df
    return output_df
    return output_df
    return output_df
    return output_df
    return output_df
    return output_df
    return output_df



data = pd.DataFrame({
    'x1': np.random.rand(100),
    'x2': np.random.rand(100),
    'y': np.random.rand(100)
})

grid = pd.DataFrame({
    'x1': np.linspace(0, 1, 10),
    'x2': np.linspace(0, 1, 10)
})

output_df = local_polynomial_quantile_regression(
    data, grid, x_vars=['x1', 'x2'], y_var='y', quantile=0.99, bandwidth=1.0, degree=1
)

print(output_df)
