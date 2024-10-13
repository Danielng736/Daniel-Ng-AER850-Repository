
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

#model 1: Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(max_iter=1000)
LR.fit(train_X, train_Y)
LR_pred = LR.predict(test_X)
LR_test = accuracy_score(LR_pred, test_Y)
print("Logistic Regression test accuracy (before best hyperparameters) is: ", round(LR_test, 5))

#model 2: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100, random_state=42)
RFC.fit(train_X, train_Y)
RFC_pred = RFC.predict(test_X)
RFC_test = accuracy_score(RFC_pred, test_Y)
print("Random Forest Classifier test accuracy (before best hyperparameters) is: ", round(RFC_test, 5))

#model 3: Support Vector Machine (SVC)
from sklearn.svm import SVC
SVM = SVC()  
SVM.fit(train_X, train_Y)
SVM_pred = SVM.predict(test_X)
SVM_test = accuracy_score(SVM_pred, test_Y)
print("Support Vector Machine Classifier test accuracy (before best hyperparameters) is: ", round(SVM_test, 5))

# Grid Search Cross Validation
# Randomized Search Cross Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

models = [
    {'name': 'LogisticRegression', 'model': LogisticRegression(random_state=42, max_iter=10000),
     'parameters': {'penalty': ['l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}},
    
    {'name': 'RandomForestClassifier', 'model': RandomForestClassifier(random_state=42),
     'parameters': {'n_estimators': [10, 30, 50, 100, 200], 'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}},
    
    {'name': 'SVC', 'model': SVC(random_state=42),
     'parameters': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto', 0.1, 1]}}
         ]

model = [
    {'name': 'RandomForestClassifier', 'model': RandomForestClassifier(random_state=42),
     'parameters': {'n_estimators': [10, 30, 50, 100, 200], 'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}}
        ]

params_array = {}
for model_params in models:
    print(f"\nOptimizing {model_params['name']} using grid seach cv...")
    grid_search = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
    grid_search.fit(train_X, train_Y)
    params = grid_search.best_params_
    params_array[model_params['name']] = params
    print(f"Best Hyperparameters for {model_params['name']}:", params)

for model_param in model:
    print(f"\nOptimizing {model_param['name']} using randomized seach cv...")
    random_search = RandomizedSearchCV(model_param['model'], model_param['parameters'], cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
    random_search.fit(train_X, train_Y)
    param = random_search.best_params_
    params_array[model_param['name']] = param
    print(f"Best Hyperparameters for {model_params['name']}:", param)
    
LR_params = params_array['LogisticRegression']
RFC_params = params_array['RandomForestClassifier']
SVM_params = params_array['SVC']

# Step 5
LR_best = LogisticRegression(max_iter=10000, **LR_params, random_state=42)
LR_best.fit(train_X, train_Y)
LR_best_predictions = LR_best.predict(test_X)

RFC_best = RandomForestClassifier(**RFC_params, random_state=42)
RFC_best.fit(train_X, train_Y)
RFC_best_predictions = RFC_best.predict(test_X)

SVM_best = SVC(**SVM_params, random_state=42)
SVM_best.fit(train_X, train_Y)
SVM_best_predictions = SVM_best.predict(test_X)

from sklearn.metrics import f1_score, precision_score, accuracy_score

def metrics_calc(predictions, true_values):
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions, average='weighted', zero_division=0)  # Considering multi-class problem
    f1 = f1_score(true_values, predictions, average='weighted')  # Considering multi-class problem
    return accuracy, precision, f1

LR_accuracy, LR_precision, LR_f1 = metrics_calc(LR_best_predictions, test_Y)
print(f"\nLogistic Regression: Accuracy: {round(LR_accuracy, 4)}, Precision: {round(LR_precision, 4)}, F1 Score: {round(LR_f1, 4)}")

RFC_accuracy, RFC_precision, RFC_f1 = metrics_calc(RFC_best_predictions, test_Y)
print(f"\nRandom Forest Classifier: Accuracy: {round(RFC_accuracy, 4)}, Precision: {round(RFC_precision, 4)}, F1 Score: {round(RFC_f1, 4)}")

SVM_accuracy, SVM_precision, SVM_f1 = metrics_calc(SVM_best_predictions, test_Y)
print(f"\nSupport Vector Machine Classifier: Accuracy: {round(SVM_accuracy, 4)}, Precision: {round(SVM_precision, 4)}, F1 Score: {round(SVM_f1, 4)}")


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y, LR_best_predictions)  
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=RFC.classes_, yticklabels=RFC.classes_) 
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Step 6
