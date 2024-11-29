'''
Created by Yuan Li on 2024-09-30

For clustering analysis of levee elevation differences.

Scatter plots of mean vs standard deviation of elevation differences for both significant and non-significant elevation differences.
Significant elevation differences: mean difference > 0.1 m.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to your data folder
data_dir = os.path.join(os.getcwd(), 'data')

# Define the path to your plots folder
plots_dir = os.path.join(os.getcwd(), 'plots')

# Create the plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Construct the full path to your CSV file
csv_path = os.path.join(data_dir, 'elevation_difference_statistics.csv')

# Read the CSV file
df = pd.read_csv(csv_path)

# Filter the DataFrame for significant and non-significant differences
df_significant = df[df['mean_diff_significant'] == 1]
df_non_significant = df[df['mean_diff_significant'] == 0]

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# First subplot (colored)
ax1.scatter(df_non_significant['mean_elevation_diff'], df_non_significant['std_elevation_diff'], 
            color='red', alpha=0.6, label='Non-significant')
ax1.scatter(df_significant['mean_elevation_diff'], df_significant['std_elevation_diff'], 
            color='blue', alpha=0.6, label='Significant')
ax1.set_xlabel('Mean Elevation Difference (m)')
ax1.set_ylabel('Standard Deviation of Elevation Difference (m)')
ax1.set_title('Colored: Mean vs Standard Deviation of Elevation Differences')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.7)

# Add text box to first subplot
num_significant = len(df_significant)
num_non_significant = len(df_non_significant)
ax1.text(0.95, 0.95, f'Significant: n = {num_significant}\nNon-significant: n = {num_non_significant}', 
         transform=ax1.transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Second subplot (black dots)
ax2.scatter(df_non_significant['mean_elevation_diff'], df_non_significant['std_elevation_diff'], 
            color='black', alpha=0.6, label='Non-significant')
ax2.scatter(df_significant['mean_elevation_diff'], df_significant['std_elevation_diff'], 
            color='black', alpha=0.6, label='Significant')
ax2.set_xlabel('Mean Elevation Difference (m)')
ax2.set_ylabel('Standard Deviation of Elevation Difference (m)')
ax2.set_title('Black: Mean vs Standard Deviation of Elevation Differences')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.7)

# Add text box to second subplot
ax2.text(0.95, 0.95, f'Significant: n = {num_significant}\nNon-significant: n = {num_non_significant}', 
         transform=ax2.transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Export the plot as PNG to the plots folder
output_path = os.path.join(plots_dir, 'elevation_filtered_difference_meanstd.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

