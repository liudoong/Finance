import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Generate 10,000 observations
num_observations = 10000

# Generate maturity values with a distribution skewed towards shorter maturities
maturity = np.random.exponential(scale=10, size=num_observations)
maturity = np.clip(maturity, 0, 50)  # Clip values to range 0 to 50 years

# Generate zspread values with a distribution skewed towards smaller spreads
zspread = np.random.exponential(scale=50, size=num_observations)
zspread = np.clip(zspread, 0, 3000)  # Clip values to range 0 to 3000 bps

# Generate return values influenced by both maturity and zspread
# Base return formula: longer maturity and higher zspread lead to higher returns
return_observation = 1 + 0.01 * maturity + 0.005 * zspread + np.random.normal(0, 0.5, num_observations)

# Create a DataFrame
df = pd.DataFrame({
    "Maturity": maturity,
    "ZSpread": zspread,
    "Return": return_observation
})

def create_panel_data(df):
    # Define bins for Duration and ZSpread
    duration_bins = np.arange(0, 36, 2).tolist() + [np.inf]
    zspread_bins = np.arange(0, 2200, 200).tolist() + [np.inf]

    # Create bins for Duration and ZSpread
    df['Duration_Bin'] = pd.cut(df['Maturity'], bins=duration_bins, right=False)
    df['ZSpread_Bin'] = pd.cut(df['ZSpread'], bins=zspread_bins, right=False)

    # Group by Duration_Bin and ZSpread_Bin
    grouped = df.groupby(['Duration_Bin', 'ZSpread_Bin'])
    
    # Calculate the 99% quantile and the count of observations for each group
    panel_data = grouped['Return'].agg([
        ('Quantile', lambda x: x.quantile(0.99)),
        ('Observation_Count', 'count')
    ]).reset_index()
    
    return panel_data

def plot_panel_data(panel_df):
    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = panel_df.pivot(index='Duration_Bin', columns='ZSpread_Bin', values='Quantile')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label': 'Quantile'})
    
    # Set labels and title
    plt.title('99% Quantile of Return by Duration and ZSpread Bins')
    plt.xlabel('ZSpread Bin')
    plt.ylabel('Duration Bin')
    
    # Show the plot
    plt.show()

# Example usage
panel_df = create_panel_data(df)
plot_panel_data(panel_df)
