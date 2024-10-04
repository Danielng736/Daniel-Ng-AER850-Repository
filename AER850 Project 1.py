# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:01:00 2024

@author: Daniel
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Project_1_Data.csv')

col1 = np.array(df['X']) 
col2 = np.array(df['Y']) 
col3 = np.array(df['Z']) 

scat_plt = plt.figure(figsize=(10, 6))
ax = scat_plt.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(col1, col2, col3, c='b', marker='o')
ax.set_title('Data Visualization: 3D Scatter Plot')
plt.show()