
"""
Created on Fri Oct  4 12:01:00 2024

@author: Daniel
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

df = pd.read_csv('Project_1_Data.csv')

col1 = np.array(df['X']) 
col2 = np.array(df['Y']) 
col3 = np.array(df['Z']) 

scat_plt = plt.figure(figsize=(10, 6))
ax = scat_plt.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

colors = np.linspace(0, 1, len(col1))

sc = ax.scatter(col1, col2, col3, c=colors, cmap='viridis', marker='o')
ax.set_title('Data Visualization: 3D Scatter Plot')

cbar = plt.colorbar(sc)
cbar.set_label('Color Scale')

plt.show()


corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='bwr', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
