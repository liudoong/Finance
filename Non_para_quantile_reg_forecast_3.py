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

# Function to replace NaN values in weights with the mean of non-NaN values
def replace_nan_with_mean(weights):
    """ Replace NaN values with the mean of the non-NaN values in each column. """
    nan_mask = np.isnan(weights)
    col_mean = np.nanmean(weights, axis=0)  # Compute column-wise mean
    weights[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])  # Replace NaN values
    return weights

# Function for local polynomial quantile regression
def local_polynomial_quantile_regression(data, grid, x_vars, y_var, quantile=0.99, bandwidth=1.0, degree=1):
    """ 
    Perform local polynomial quantile regression using different kernel functions.
    """
    # Extract data points and grid points
    grid_points = grid[x_vars].values
    data_points = data[x_vars].values
    y = data[y_var].values
    
    # Compute distances between data points and grid points
    distances = cdist(data_points, grid_points, metric='euclidean')
    
    # Compute kernel weights
    weights_tri = tri_cube_kernel(distances / bandwidth)
    weights_gauss = gaussian_kernel(distances / bandwidth)
    weights_epan = epanechnikov_kernel(distances / bandwidth)
    
    # Replace NaN values in weights with the mean of non-NaN values
    weights_tri = replace_nan_with_mean(weights_tri)
    weights_gauss = replace_nan_with_mean(weights_gauss)
    weights_epan = replace_nan_with_mean(weights_epan)
    
    # Count neighborhood points used for each kernel
    num_neighborhood_tri = np.sum(weights_tri > 0, axis=0)
    num_neighborhood_gauss = np.sum(weights_gauss > 0, axis=0)
    num_neighborhood_epan = np.sum(weights_epan > 0, axis=0)
    
    # Normalize weights to ensure they sum to 1 across all data points for each grid point
    weights_tri /= np.sum(weights_tri, axis=0, keepdims=True)
    weights_gauss /= np.sum(weights_gauss, axis=0, keepdims=True)
    weights_epan /= np.sum(weights_epan, axis=0, keepdims=True)
    
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
    
    # Store predictions and neighborhood point counts in a DataFrame
    output_df = pd.DataFrame({
        'TriCube_Predicted': predicted_tri,
        'TriCube_Neighborhood_Points': num_neighborhood_tri,
        'Gaussian_Predicted': predicted_gauss,
        'Gaussian_Neighborhood_Points': num_neighborhood_gauss,
        'Epanechnikov_Predicted': predicted_epan,
        'Epanechnikov_Neighborhood_Points': num_neighborhood_epan
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




