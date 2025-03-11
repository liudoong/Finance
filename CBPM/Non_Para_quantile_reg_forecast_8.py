import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.regression.quantile_regression import QuantReg

def generate_sample_data():
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    # Generate dates
    start_date = datetime(2000, 1, 1)
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
            zsprd_m = np.random.uniform(1, 1500)
            data.append([date.strftime('%Y%m%d'), isin, None, price, moddur_m, zsprd_m])
        for cusip in cusips:
            price = np.random.uniform(90, 110)
            moddur_m = np.random.uniform(1, 10)
            zsprd_m = np.random.uniform(1, 1500)
            data.append([date.strftime('%Y%m%d'), None, cusip, price, moddur_m, zsprd_m])
    # Create DataFrame
    columns = ['Date', 'ISIN', 'CUSIP', 'PRICE', 'MODDUR_M', 'ZSPRD_M']
    df = pd.DataFrame(data, columns=columns)
    return df

def generate_grid_data(moddur_range, zsprd_range, num_points):
    """
    Generate a grid of MODDUR_M and ZSPRD_M values for quantile regression.
    """
    moddur_values = np.linspace(moddur_range[0], moddur_range[1], num_points)
    zsprd_values = np.linspace(zsprd_range[0], zsprd_range[1], num_points)
    moddur_grid, zsprd_grid = np.meshgrid(moddur_values, zsprd_values)
    moddur_grid = moddur_grid.flatten()
    zsprd_grid = zsprd_grid.flatten()
    grid_data = pd.DataFrame({
        'MODDUR_M': moddur_grid,
        'ZSPRD_M': zsprd_grid
    })
    return grid_data

def dataset(df, identifier, mpr):
    """
    Preprocess the dataset based on the identifier (ISIN or CUSIP) and calculate the return (RT).
    """
    if identifier == 'ISIN':
        df = df[['Date', 'ISIN', 'PRICE', 'MODDUR_M', 'ZSPRD_M']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
        df.set_index(['Date', 'ISIN'], inplace=True)
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
    """ Tri-cube kernel function used for weighting. """
    return np.where(np.abs(u) <= 1, (1 - np.abs(u)**3)**3, 0)

def gaussian_kernel(u):
    """ Gaussian kernel function used for weighting. """
    return np.exp(-0.5 * u**2)

def uniform_kernel(u):
    """ Uniform kernel function used for weighting. """
    return np.where(np.abs(u) <= 1, 1, 0)

def local_polynomial_quantile_regression_for_grid(
    df, grid_data, kernel_func, bandwidth=1, degree=1, quantile=0.01, num_neighbors=None):
    """
    Perform local polynomial quantile regression using statsmodels.QuantReg with weights.
    """
    if kernel_func == 'tri_cube':
        kernel = tri_cube_kernel
    elif kernel_func == 'gaussian':
        kernel = gaussian_kernel
    elif kernel_func == 'uniform':
        kernel = uniform_kernel
    else:
        raise ValueError("Invalid kernel function. Choose from 'tri_cube', 'gaussian', or 'uniform'.")

    X = df[['MODDUR_M', 'ZSPRD_M']].values
    y = df['RT'].values
    X_grid = grid_data[['MODDUR_M', 'ZSPRD_M']].values

    forecasted_rt_grid = np.zeros(len(grid_data))
    regression_results = []

    for i in range(len(X_grid)):
        distances = np.linalg.norm(X - X_grid[i], axis=1)
        if num_neighbors is not None:
            nearest_indices = np.argsort(distances)[:num_neighbors]
            distances = distances[nearest_indices]
            X_nearest = X[nearest_indices]
            y_nearest = y[nearest_indices]
        else:
            X_nearest = X
            y_nearest = y

        weights = kernel(distances / bandwidth)
        X_poly = np.column_stack([X_nearest[:, 0]**d for d in range(degree + 1)] +
                                 [X_nearest[:, 1]**d for d in range(degree + 1)])

        # Apply weights to the data
        sqrt_weights = np.sqrt(weights)
        y_weighted = y_nearest * sqrt_weights
        X_weighted = X_poly * sqrt_weights[:, np.newaxis]

        # Fit the quantile regression model using statsmodels.QuantReg
        model = QuantReg(y_weighted, X_weighted)
        results = model.fit(q=quantile)
        
        X_poly_grid = np.column_stack([X_grid[i, 0]**d for d in range(degree + 1)] +
                                      [X_grid[i, 1]**d for d in range(degree + 1)])
        forecasted_rt_grid[i] = results.predict(X_poly_grid)

        regression_results.append({
            'coefficients': results.params,
            'bandwidth': bandwidth,
            'degree': degree,
            'quantile': quantile
        })
    grid_results = grid_data.copy()
    grid_results['Forecasted_RT'] = forecasted_rt_grid
    return grid_results, regression_results

def forecast_out_of_sample(new_data, regression_results):
    """
    Forecast RT for out-of-sample data points using the saved regression results.
    """
    X_new = new_data[['MODDUR_M', 'ZSPRD_M']].values
    forecasted_rt_new = np.zeros(len(new_data))

    for i in range(len(new_data)):
        results = regression_results[0]  # Using the first grid point's results for simplicity
        coefficients = results['coefficients']
        degree = results['degree']
        X_poly_new = np.column_stack([X_new[i, 0]**d for d in range(degree + 1)] +
                                     [X_new[i, 1]**d for d in range(degree + 1)])
        forecasted_rt_new[i] = np.dot(X_poly_new, coefficients)

    return forecasted_rt_new




# Define the range and number of points for MODDUR_M and ZSPRD_M
moddur_range = (1, 10)
zsprd_range = (100, 1000)
num_points = 10

# Generate the grid data
grid_data = generate_grid_data(moddur_range, zsprd_range, num_points)
data = generate_sample_data()
data_set = dataset(data, 'CUSIP', 5)

# Perform local polynomial quantile regression
grid_results, regression_results = local_polynomial_quantile_regression_for_grid(
    data_set, grid_data, kernel_func='gaussian', bandwidth=1.0, degree=2, quantile=0.01
)

# Create new out-of-sample data points
new_data = pd.DataFrame({
    'MODDUR_M': [5.5, 6.5, 7.5],
    'ZSPRD_M': [1.55, 1.65, 1.75]
})

# Forecast RT for the new data points
forecasted_rt_new = forecast_out_of_sample(new_data, regression_results)
new_data['Forecasted_RT'] = forecasted_rt_new

# Display the new data with forecasted RT
print(new_data)