
"""
Created on Fri Oct  4 12:01:00 2024

@author: Daniel
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Project_1_Data.csv')

# Step 2
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

# Step 3
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='bwr', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Step 4
X = df[['X', 'Y', 'Z']]
Y = df['Step']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

#model 1 Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000)
LR.fit(train_X, train_Y)
LR_pred = LR.predict(test_X)
LR_test = accuracy_score(LR_pred, test_Y)
print("Logistic Regression test accuracy (before best hyperparameters) is: ", round(LR_test, 5))

#model 2 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100, random_state=42)
RFC.fit(train_X, train_Y)
RFC_pred = RFC.predict(test_X)
RFC_test = accuracy_score(RFC_pred, test_Y)
print("Random Forest Classifier test accuracy (before best hyperparameters) is: ", round(RFC_test, 5))

#model 3 SVM (SVC)
from sklearn.svm import SVC
SVM = SVC()  
SVM.fit(train_X, train_Y)
SVM_pred = SVM.predict(test_X)
SVM_test = accuracy_score(SVM_pred, test_Y)
print("Support Vector Machine Classifier test accuracy (before best hyperparameters) is: ", round(SVM_test, 5))