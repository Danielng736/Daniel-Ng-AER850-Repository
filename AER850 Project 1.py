# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:01:00 2024

@author: Daniel
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# Load your data
df = pd.read_csv('Project_1_Data.csv')

# Extract the columns
col1 = np.array(df['X']) 
col2 = np.array(df['Y']) 
col3 = np.array(df['Z']) 

# Create a figure for the 3D scatter plot
scat_plt = plt.figure(figsize=(10, 6))
ax = scat_plt.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Create a color array based on the length of your data
colors = np.linspace(0, 1, len(col1))

# Scatter plot
sc = ax.scatter(col1, col2, col3, c=colors, cmap='viridis', marker='o')
ax.set_title('Data Visualization: 3D Scatter Plot')

# Add a color bar
cbar = plt.colorbar(sc)
cbar.set_label('Color Scale')

# Show the plot
plt.show()
