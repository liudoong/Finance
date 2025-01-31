import os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.linear_model import QuantileRegressor

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
np.random.seed(42)  # For reproducibility

# Generate dates
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# Generate ISINs and CUSIPs
isins = ['ISIN001', 'ISIN002', 'ISIN003']
cusips = ['CUSIP001', 'CUSIP002', 'CUSIP003']

# Generate random prices, modified duration, and Z-spread
data = []
for date in dates:
    for isin in isins:
        price = np.random.uniform(90, 110)
        moddur_m = np.random.uniform(1, 10)
        zsprd_m = np.random.uniform(0.5, 2.5)
        data.append([date.strftime('%Y%m%d'), isin, None, price, moddur_m, zsprd_m])
    
    for cusip in cusips:
        price = np.random.uniform(90, 110)
        moddur_m = np.random.uniform(1, 10)
        zsprd_m = np.random.uniform(0.5, 2.5)
        data.append([date.strftime('%Y%m%d'), None, cusip, price, moddur_m, zsprd_m])

# Create DataFrame
columns = ['Date', 'ISIN', 'CUSIP', 'PRICE', 'MODDUR_M', 'ZSPRD_M']
df = pd.DataFrame(data, columns=columns)


def generate_grid_data(moddur_range, zsprd_range, num_points):
    """
    Generate a grid of MODDUR_M and ZSPRD_M values for quantile regression.

    Parameters:
    moddur_range (tuple): A tuple (min, max) specifying the range of MODDUR_M values.
    zsprd_range (tuple): A tuple (min, max) specifying the range of ZSPRD_M values.
    num_points (int): The number of points to generate in each dimension.

    Returns:
    pd.DataFrame: A DataFrame containing the grid of MODDUR_M and ZSPRD_M values.
    """
    # Generate evenly spaced values for MODDUR_M and ZSPRD_M
    moddur_values = np.linspace(moddur_range[0], moddur_range[1], num_points)
    zsprd_values = np.linspace(zsprd_range[0], zsprd_range[1], num_points)

    # Create a grid of MODDUR_M and ZSPRD_M values
    moddur_grid, zsprd_grid = np.meshgrid(moddur_values, zsprd_values)
    moddur_grid = moddur_grid.flatten()
    zsprd_grid = zsprd_grid.flatten()

    # Create a DataFrame from the grid
    grid_data = pd.DataFrame({
        'MODDUR_M': moddur_grid,
        'ZSPRD_M': zsprd_grid
    })

    return grid_data

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


def local_polynomial_quantile_regression_for_grid(df, grid_data, kernel_func, bandwidth, degree, quantile=0.5):
    """
    Perform local polynomial quantile regression on a large dataset but only calculate forecasts for grid points.

    Parameters:
    df (pd.DataFrame): The large dataset with columns ['MODDUR_M', 'ZSPRD_M', 'RT'].
    grid_data (pd.DataFrame): The grid data with columns ['MODDUR_M', 'ZSPRD_M'].
    kernel_func (str): The kernel function to use ('tri_cube', 'gaussian', or 'epanechnikov').
    bandwidth (float): The bandwidth for the kernel function.
    degree (int): The degree of the polynomial.
    quantile (float): The quantile to regress (default is 0.5 for median regression).

    Returns:
    pd.DataFrame: A DataFrame with the grid points and their forecasted RT values.
    dict: A dictionary containing the regression results for future use.
    """
    # Define the kernel function based on the input
    if kernel_func == 'tri_cube':
        kernel = tri_cube_kernel
    elif kernel_func == 'gaussian':
        kernel = gaussian_kernel
    elif kernel_func == 'epanechnikov':
        kernel = epanechnikov_kernel
    else:
        raise ValueError("Invalid kernel function. Choose from 'tri_cube', 'gaussian', or 'epanechnikov'.")

    # Extract the features (MODDUR_M, ZSPRD_M) and target (RT) from the large dataset
    X = df[['MODDUR_M', 'ZSPRD_M']].values
    y = df['RT'].values

    # Extract the grid points
    X_grid = grid_data[['MODDUR_M', 'ZSPRD_M']].values

    # Initialize arrays to store the forecasted RT for grid points and regression results
    forecasted_rt_grid = np.zeros(len(grid_data))
    regression_results = []

    # Loop through each grid point
    for i in range(len(X_grid)):
        # Calculate the distance between the grid point and all points in the large dataset
        distances = np.linalg.norm(X - X_grid[i], axis=1)
        
        # Calculate the weights using the kernel function
        weights = kernel(distances / bandwidth)
        
        # Create polynomial features for the large dataset
        X_poly = np.column_stack([X[:, 0]**d for d in range(degree + 1)] +
                                 [X[:, 1]**d for d in range(degree + 1)])
        
        # Fit the quantile regression model
        model = QuantileRegressor(quantile=quantile, alpha=0.0)
        model.fit(X_poly, y, sample_weight=weights)
        
        # Create polynomial features for the grid point
        X_poly_grid = np.column_stack([X_grid[i, 0]**d for d in range(degree + 1)] +
                                      [X_grid[i, 1]**d for d in range(degree + 1)])
        
        # Forecast the RT for the grid point
        forecasted_rt_grid[i] = model.predict(X_poly_grid)
        
        # Store the regression results for future use
        regression_results.append({
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'bandwidth': bandwidth,
            'degree': degree,
            'quantile': quantile
        })

    # Add the forecasted RT to the grid data
    grid_data['Forecasted_RT'] = forecasted_rt_grid

    return grid_data, regression_results

def forecast_out_of_sample(new_data, regression_results):
    """
    Forecast RT for out-of-sample data points using the saved regression results.

    Parameters:
    new_data (pd.DataFrame): A DataFrame containing new data points with columns ['MODDUR_M', 'ZSPRD_M'].
    regression_results (list): A list of dictionaries containing regression results from the previous function.

    Returns:
    np.array: An array of forecasted RT values for the new data points.
    """
    # Extract the new features (MODDUR_M, ZSPRD_M)
    X_new = new_data[['MODDUR_M', 'ZSPRD_M']].values

    # Initialize an array to store the forecasted RT values
    forecasted_rt_new = np.zeros(len(new_data))

    # Loop through each new data point
    for i in range(len(new_data)):
        # Find the nearest grid point (for simplicity, use the first regression result)
        # In practice, you might want to use a more sophisticated method to select the appropriate regression results
        results = regression_results[0]  # Using the first grid point's results for simplicity

        # Extract the coefficients, intercept, and degree
        coefficients = results['coefficients']
        intercept = results['intercept']
        degree = results['degree']

        # Create polynomial features for the new point
        X_poly_new = np.column_stack([X_new[i, 0]**d for d in range(degree + 1)] +
                                     [X_new[i, 1]**d for d in range(degree + 1)])

        # Forecast the RT for the new point
        forecasted_rt_new[i] = np.dot(X_poly_new, coefficients) + intercept

    return forecasted_rt_new

# Assuming df is the large dataset and grid_data is the smaller grid
grid_data, regression_results = local_polynomial_quantile_regression_for_grid(
    df, grid_data, kernel_func='gaussian', bandwidth=1.0, degree=2, quantile=0.5
)

# Display the grid data with forecasted RT
print(grid_data.head())

# Create new out-of-sample data points
new_data = pd.DataFrame({
    'MODDUR_M': [5.5, 6.5, 7.5],
    'ZSPRD_M': [1.55, 1.65, 1.75]
})

# Forecast RT for the new data points
forecasted_rt_new = forecast_out_of_sample(new_data, regression_results)

# Add the forecasted RT to the new data
new_data['Forecasted_RT'] = forecasted_rt_new

# Display the new data with forecasted RT
print(new_data)