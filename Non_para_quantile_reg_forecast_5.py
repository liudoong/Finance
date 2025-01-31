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

def local_polynomial_quantile_regression(df, kernel_func, bandwidth, degree, quantile=0.5):
    """
    Perform local polynomial quantile regression on the dataset.

    Parameters:
    df (pd.DataFrame): The preprocessed dataset from the `dataset` function.
    kernel_func (str): The kernel function to use ('tri_cube', 'gaussian', or 'epanechnikov').
    bandwidth (float): The bandwidth for the kernel function.
    degree (int): The degree of the polynomial.
    quantile (float): The quantile to regress (default is 0.5 for median regression).

    Returns:
    pd.DataFrame: A DataFrame with the forecasted RT values.
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

    # Extract the features (MODDUR_M, ZSPRD_M) and target (RT)
    X = df[['MODDUR_M', 'ZSPRD_M']].values
    y = df['RT'].values

    # Initialize arrays to store the forecasted RT and regression results
    forecasted_rt = np.zeros(len(df))
    regression_results = []

    # Loop through each point in the dataset
    for i in range(len(df)):
        # Calculate the distance between the current point and all other points
        distances = np.linalg.norm(X - X[i], axis=1)
        
        # Calculate the weights using the kernel function
        weights = kernel(distances / bandwidth)
        
        # Create polynomial features
        X_poly = np.column_stack([X[:, 0]**d for d in range(degree + 1)] +
                                 [X[:, 1]**d for d in range(degree + 1)])
        
        # Fit the quantile regression model
        model = QuantileRegressor(quantile=quantile, alpha=0.0)
        model.fit(X_poly, y, sample_weight=weights)
        
        # Forecast the RT for the current point
        X_poly_i = np.column_stack([X[i, 0]**d for d in range(degree + 1)] +
                                   [X[i, 1]**d for d in range(degree + 1)])
        forecasted_rt[i] = model.predict(X_poly_i)
        
        # Store the regression results for future use
        regression_results.append({
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'bandwidth': bandwidth,
            'degree': degree,
            'quantile': quantile
        })

    # Add the forecasted RT to the DataFrame
    df['Forecasted_RT'] = forecasted_rt

    return df, regression_results

# Assuming regression_results is the output from the `local_polynomial_quantile_regression` function

# Create new data points
new_data = pd.DataFrame({
    'MODDUR_M': [5.0, 6.0, 7.0],
    'ZSPRD_M': [1.5, 1.6, 1.7]
})

# Forecast RT for the new data points
forecasted_rt_new = forecast_new_points(new_data, regression_results)

# Display the forecasted RT values
print(forecasted_rt_new)