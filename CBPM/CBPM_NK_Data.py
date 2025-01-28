def dataset(df, identifier, mpr):
    if identifier =="ISIN":
        df = df[['Date','ISIN','PRICE','MODDUR_M','ZSPRD_M']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], formate = '%Y%m%d').dt.date
        df.set_index(['Date','ISIN'], inplace = True)
        df['RT'] = df.groupby('ISIN')['PRICE'].pct_change(mpr)
        df = df.dropna()
    elif: identifier == 'CUSIP':
        df = df[['Date','CUSIP','PRICE','MODDUR_M','ZSPRD_M']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], formate = '%Y%m%d').dt.date
        df.set_index(['Date','CUSIP'], inplace = True)
        df['RT'] = df.groupby('CUSIP')['PRICE'].pct_change(mpr)
    return df 

def tri_cube_kernel(u):
    return np.where(np.abs(u) <=1, (1-np.abs(u)**3)**3, 0)

def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

def epanechnikov_kernel(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
    
def local_polynomial_quantile_regression(data, x_vars, y_var, quantile = 0.99, bandwidth = 1.0, degree = 1):
    results = []
    for i, row in data.iterrows():

        x0 = row[x_vars].values

        distances = nplinalg.norm(data[x_vars] - x0, axis = 1)

        weights = tri_cube_kernel(distances / bandwidth)

        x = np.column_stack([data[x_vars]**d for d in range(degree + 1)])
        y = data[y_var]

        model = QuantReg(y,X)
        result = model.fit(q = quantile, weights = weights)

        x0_ploy = np.column_stack([x0**d for d in range(degree + 1)])

        results. append(result.predict(x0_poly.reshape(1,-1))[0])

    return results