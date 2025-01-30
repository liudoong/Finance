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
        dict: A dictionary containing precomputed information for making predictions on new points.
    """
    # Extract grid points and data points as numpy arrays
    grid_points = grid[x_vars].values  # Shape: (n_grid_points, n_x_vars)
    data_points = data[x_vars].values  # Shape: (n_data_points, n_x_vars)
    y = data[y_var].values             # Shape: (n_data_points,)
    
    # Compute distances between all data points and grid points
    distances = np.linalg.norm(data_points[:, np.newaxis, :] - grid_points[np.newaxis, :, :], axis=2)  # Shape: (n_data_points, n_grid_points)
    
    # Compute weights for each kernel
    weights_tri = tri_cube_kernel(distances / bandwidth)  # Shape: (n_data_points, n_grid_points)
    weights_gauss = gaussian_kernel(distances / bandwidth)  # Shape: (n_data_points, n_grid_points)
    weights_epan = epanechnikov_kernel(distances / bandwidth)  # Shape: (n_data_points, n_grid_points)
    
    # Compute neighborhood points for each kernel
    num_neighborhood_tri = np.sum(weights_tri > 0, axis=0)  # Shape: (n_grid_points,)
    num_neighborhood_gauss = np.sum(weights_gauss > 0, axis=0)  # Shape: (n_grid_points,)
    num_neighborhood_epan = np.sum(weights_epan > 0, axis=0)  # Shape: (n_grid_points,)
    
    # Construct design matrix for data points
    X_data = np.column_stack([data_points**d for d in range(degree + 1)])  # Shape: (n_data_points, degree + 1)
    
    # Construct design matrix for grid points
    X_grid = np.column_stack([grid_points**d for d in range(degree + 1)])  # Shape: (n_grid_points, degree + 1)
    
    # Initialize arrays to store predictions
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
        
        # Epanechnikov kernel
        model_epan = QuantReg(y, X_data)
        result_epan = model_epan.fit(q=quantile, weights=weights_epan[:, i])
        predicted_epan[i] = result_epan.predict(X_grid[i].reshape(1, -1))[0]
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'TriCube_Predicted': predicted_tri,
        'TriCube_Neighborhood_Points': num_neighborhood_tri,
        'Gaussian_Predicted': predicted_gauss,
        'Gaussian_Neighborhood_Points': num_neighborhood_gauss,
        'Epanechnikov_Predicted': predicted_epan,
        'Epanechnikov_Neighborhood_Points': num_neighborhood_epan
    })
    
    # Store precomputed information for predictions
    precomputed_info = {
        'X_data': X_data,  # Design matrix for data points
        'y': y,            # Target variable
        'data_points': data_points,  # Original data points
        'bandwidth': bandwidth,
        'degree': degree,
        'quantile': quantile
    }
    
    return output_df, precomputed_info


def predict_new_points(new_points, precomputed_info):
    """
    Predict values for new points using precomputed information from local polynomial quantile regression.

    Parameters:
        new_points (pd.DataFrame): New points to predict, with columns matching the original x_vars.
        precomputed_info (dict): Precomputed information from the regression function.

    Returns:
        pd.DataFrame: A DataFrame containing predictions for the new points.
    """
    # Extract precomputed information
    X_data = precomputed_info['X_data']
    y = precomputed_info['y']
    data_points = precomputed_info['data_points']
    bandwidth = precomputed_info['bandwidth']
    degree = precomputed_info['degree']
    quantile = precomputed_info['quantile']
    
    # Extract new points as numpy array
    new_points = new_points.values  # Shape: (n_new_points, n_x_vars)
    
    # Compute distances between new points and data points
    distances = np.linalg.norm(data_points[:, np.newaxis, :] - new_points[np.newaxis, :, :], axis=2)  # Shape: (n_data_points, n_new_points)
    
    # Compute weights for each kernel
    weights_tri = tri_cube_kernel(distances / bandwidth)  # Shape: (n_data_points, n_new_points)
    weights_gauss = gaussian_kernel(distances / bandwidth)  # Shape: (n_data_points, n_new_points)
    weights_epan = epanechnikov_kernel(distances / bandwidth)  # Shape: (n_data_points, n_new_points)

    # Construct design matrix for new points
    X_new = np.column_stack([new_points**d for d in range(degree + 1)])  # Shape: (n_new_points, degree + 1)
    
    # Initialize arrays to store predictions
    predicted_tri = np.zeros(new_points.shape[0])  # Shape: (n_new_points,)
    predicted_gauss = np.zeros(new_points.shape[0])  # Shape: (n_new_points,)
    predicted_epan = np.zeros(new_points.shape[0])  # Shape: (n_new_points,)
    
    # Fit models and make predictions for each new point
    for i in range(new_points.shape[0]):
        # Tri-cube kernel
        model_tri = QuantReg(y, X_data)
        result_tri = model_tri.fit(q=quantile, weights=weights_tri[:, i])
        predicted_tri[i] = result_tri.predict(X_new[i].reshape(1, -1))[0]
        
        # Gaussian kernel
        model_gauss = QuantReg(y, X_data)
        result_gauss = model_gauss.fit(q=quantile, weights=weights_gauss[:, i])
        predicted_gauss[i] = result_gauss.predict(X_new[i].reshape(1, -1))[0]
        
        # Epanechnikov kernel
        model_epan = QuantReg(y, X_data)
        result_epan = model_epan.fit(q=quantile, weights=weights_epan[:, i])
        predicted_epan[i] = result_epan.predict(X_new[i].reshape(1, -1))[0]
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'TriCube_Predicted': predicted_tri,
        'Gaussian_Predicted': predicted_gauss,
        'Epanechnikov_Predicted': predicted_epan
    })
    
    return output_df


# Example data
data = pd.DataFrame({
    'MODDUR_M': np.random.rand(100),  # Independent variable 1
    'ZSPRD_M': np.random.rand(100),   # Independent variable 2
    'RT': np.random.rand(100)         # Dependent variable
})

# Grid points for initial regression
grid = pd.DataFrame({
    'MODDUR_M': np.linspace(0, 1, 10),  # Grid points for MODDUR_M
    'ZSPRD_M': np.linspace(0, 1, 10)    # Grid points for ZSPRD_M
})

# Perform initial regression and get precomputed information
results, precomputed_info = local_polynomial_quantile_regression(
    data, grid, x_vars=['MODDUR_M', 'ZSPRD_M'], y_var='RT', quantile=0.99, bandwidth=1.0, degree=1
)

# New points to predict
new_points = pd.DataFrame({
    'MODDUR_M': [0.5, 0.6, 0.7],  # New MODDUR_M values
    'ZSPRD_M': [0.3, 0.4, 0.5]    # New ZSPRD_M values
})

# Make predictions for the new points
predictions = predict_new_points(new_points, precomputed_info)

# Display predictions
print(predictions)