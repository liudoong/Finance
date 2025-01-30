import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.linear_model import QuantileRegressor

# Define kernel functions used for weighting nearby points
def tri_cube_kernel(u):
    """ Tri-cube kernel function used for weighting. """
    return np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0)

def gaussian_kernel(u):
    """ Gaussian kernel function used for weighting. """
    return np.exp(-0.5 * u**2)

def epanechnikov_kernel(u):
    """ Epanechnikov kernel function used for weighting. """
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)

# Function for local polynomial quantile regression
def local_polynomial_quantile_regression(data, grid, x_vars, y_var, quantile=0.99, bandwidth=1.0, degree=1):
    """ 
    Perform local polynomial quantile regression using different kernel functions.
    
    Parameters:
        data (pd.DataFrame): The dataset containing independent and dependent variables.
        grid (pd.DataFrame): Grid points where predictions will be made.
        x_vars (list): List of independent variable names.
        y_var (str): Name of the dependent variable.
        quantile (float): The quantile to estimate (default 0.99 for upper quantile).
        bandwidth (float): Bandwidth for kernel functions (default 1.0).
        degree (int): Degree of the polynomial regression (default 1 for linear regression).
    
    Returns:
        pd.DataFrame: DataFrame containing predictions for different kernel functions.
        dict: Precomputed model information for future predictions.
    """
    # Extract data points and grid points
    grid_points = grid[x_vars].values
    data_points = data[x_vars].values
    y = data[y_var].values
    
    # Compute distances between data points and grid points
    distances = cdist(data_points, grid_points, metric='euclidean')
    
    # Compute kernel weights for each distance matrix
    weights_tri = tri_cube_kernel(distances / bandwidth)
    weights_gauss = gaussian_kernel(distances / bandwidth)
    weights_epan = epanechnikov_kernel(distances / bandwidth)
    
    # Normalize weights to ensure they sum to 1 across all data points for each grid point
    weights_tri /= weights_tri.sum(axis=0, keepdims=True)
    weights_gauss /= weights_gauss.sum(axis=0, keepdims=True)
    weights_epan /= weights_epan.sum(axis=0, keepdims=True)
    
    # Construct polynomial features for regression
    X_data = np.column_stack([data_points**d for d in range(degree + 1)])
    X_grid = np.column_stack([grid_points**d for d in range(degree + 1)])
    
    # Initialize arrays to store predictions
    predicted_tri = np.zeros(grid_points.shape[0])
    predicted_gauss = np.zeros(grid_points.shape[0])
    predicted_epan = np.zeros(grid_points.shape[0])
    
    # Function to fit quantile regression using sklearn's QuantileRegressor
    def fit_quantile_regression(X, y, weights, quantile):
        model = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
        model.fit(X, y, sample_weight=weights)
        return model
    
    # Train and predict for each kernel function at each grid point
    for i in range(grid_points.shape[0]):
        model_tri = fit_quantile_regression(X_data, y, weights_tri[:, i], quantile)
        model_gauss = fit_quantile_regression(X_data, y, weights_gauss[:, i], quantile)
        model_epan = fit_quantile_regression(X_data, y, weights_epan[:, i], quantile)
        
        predicted_tri[i] = model_tri.predict(X_grid[i].reshape(1, -1))
        predicted_gauss[i] = model_gauss.predict(X_grid[i].reshape(1, -1))
        predicted_epan[i] = model_epan.predict(X_grid[i].reshape(1, -1))
    
    # Store predictions in a DataFrame
    output_df = pd.DataFrame({
        'TriCube_Predicted': predicted_tri,
        'Gaussian_Predicted': predicted_gauss,
        'Epanechnikov_Predicted': predicted_epan
    })
    
    # Store precomputed information for further predictions
    precomputed_info = {
        'X_data': X_data,
        'y': y,
        'data_points': data_points,
        'bandwidth': bandwidth,
        'degree': degree,
        'quantile': quantile
    }
    
    return output_df, precomputed_info

# Function to predict new data points using precomputed model
def predict_new_points(new_points, precomputed_info):
    """
    Predict values for new data points using precomputed regression model.
    
    Parameters:
        new_points (pd.DataFrame): DataFrame containing new points to predict.
        precomputed_info (dict): Precomputed model information.
    
    Returns:
        pd.DataFrame: Predictions for the new data points.
    """
    X_data = precomputed_info['X_data']
    y = precomputed_info['y']
    degree = precomputed_info['degree']
    quantile = precomputed_info['quantile']
    
    new_points = new_points.values
    X_new = np.column_stack([new_points**d for d in range(degree + 1)])
    
    model = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
    model.fit(X_data, y)
    predictions = model.predict(X_new)
    
    return pd.DataFrame({'Predictions': predictions})

# Example usage
data = pd.DataFrame({
    'MODDUR_M': np.random.rand(100),
    'ZSPRD_M': np.random.rand(100),
    'RT': np.random.rand(100)
})

grid = pd.DataFrame({
    'MODDUR_M': np.linspace(0, 1, 10),
    'ZSPRD_M': np.linspace(0, 1, 10)
})

# Perform quantile regression and store results
results, precomputed_info = local_polynomial_quantile_regression(
    data, grid, x_vars=['MODDUR_M', 'ZSPRD_M'], y_var='RT', quantile=0.99, bandwidth=1.0, degree=1
)

# Predict new points
new_points = pd.DataFrame({
    'MODDUR_M': [0.5, 0.6, 0.7],
    'ZSPRD_M': [0.3, 0.4, 0.5]
})

predictions = predict_new_points(new_points, precomputed_info)
print(predictions)




