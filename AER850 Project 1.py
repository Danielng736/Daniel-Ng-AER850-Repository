"""
Created on Fri Oct  4 12:01:00 2024

@author: Daniel Ng
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import StackingClassifier
import joblib

df = pd.read_csv('Project_1_Data.csv')

# Step 2
# Defined each column using numpy
col1 = np.array(df['X']) 
col2 = np.array(df['Y']) 
col3 = np.array(df['Z']) 

# Scatter plot being used
scat_plt = plt.figure(figsize=(10, 6))
ax = scat_plt.add_subplot(111, projection='3d')

# Set the x, y, z labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

colors = np.linspace(0, 1, len(col1))

# Sets the axis and the title for the scatter plot
sc = ax.scatter(col1, col2, col3, c=colors, cmap='viridis', marker='o')
ax.set_title('3D Scatter Plot of Data Visualization')

# Plots the colorbar for the scatter plot
cbar = plt.colorbar(sc)

plt.show()

# Step 3
# Defined the correlation matrix
corr_matrix = df.corr()

# PLots the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='bwr', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Step 4
X = df[['X', 'Y', 'Z']]
Y = df['Step']

# Trains the X and Y as well as well as testing them using 20% testing size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Logistic Regression
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train, Y_train)
LR_pred = LR.predict(X_test)
LR_test = accuracy_score(LR_pred, Y_test)
print("\nLogistic Regression test accuracy is: ", round(LR_test, 4))

# Random Forest Classifier
RFC = RandomForestClassifier(n_estimators=100, random_state=42)
RFC.fit(X_train, Y_train)
RFC_pred = RFC.predict(X_test)
RFC_test = accuracy_score(RFC_pred, Y_test)
print("\nRandom Forest Classifier test accuracy is: ", round(RFC_test, 4))

# Support Vector Machine
SVM = SVC()  
SVM.fit(X_train, Y_train)
SVM_pred = SVM.predict(X_test)
SVM_test = accuracy_score(SVM_pred, Y_test)
print("\nSupport Vector Machine Classifier test accuracy is: ", round(SVM_test, 4))

# Model 1 parameters
model_1 = [
    {'name': 'LogisticRegression', 'model': LogisticRegression(random_state=42, max_iter=10000),
     'parameters': {'penalty': ['l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}},
    
    {'name': 'RandomForestClassifier', 'model': RandomForestClassifier(random_state=42),
     'parameters': {'n_estimators': [10, 30, 50, 100, 200], 'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}},
    
    {'name': 'SVC', 'model': SVC(random_state=42),
     'parameters': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto', 0.1, 1]}}
         ]

# Model 2 parameters
model_2 = [
    {'name': 'RandomForestClassifier', 'model': RandomForestClassifier(random_state=42),
     'parameters': {'n_estimators': [10, 30, 50, 100, 200], 'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}}
          ]

# Using GridSearchCV() to cross validate the models and looks for optimized parameters
params_array_1 = {}
for model_params in model_1:
    grid_search = GridSearchCV(model_params['model'], model_params['parameters'], cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
    grid_search.fit(X_train, Y_train)
    params = grid_search.best_params_
    params_array_1[model_params['name']] = params
    print(f"\nBest Hyperparameters for {model_params['name']}:", params)
    
print('\ndone')

# Using RandomizedSearchCV() on RandomForestClassifier as another way for optimizing results
params_array_2 = {}
for model_param in model_2:
    random_search = RandomizedSearchCV(model_param['model'], model_param['parameters'], cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
    random_search.fit(X_train, Y_train)
    param = random_search.best_params_
    params_array_2[model_param['name']] = param
    print(f"\nBest Hyperparameters for {model_param['name']}:", param)

# Storing LR, RFC and SVM optimized parameters 
LR_params = params_array_1['LogisticRegression']
RFC_params_1 = params_array_1['RandomForestClassifier']
SVM_params = params_array_1['SVC']
RFC_params_2 = params_array_2['RandomForestClassifier']

# Step 5
# Using the optimizd parameters in LR, RFC and SVC
LR_best = LogisticRegression(max_iter=10000, **LR_params, random_state=42)
LR_best.fit(X_train, Y_train)
LR_best_pred = LR_best.predict(X_test)

RFC_best_1 = RandomForestClassifier(**RFC_params_1, random_state=42)
RFC_best_1.fit(X_train, Y_train)
RFC_best_pred_1 = RFC_best_1.predict(X_test)

RFC_best_2 = RandomForestClassifier(**RFC_params_2, random_state=42)
RFC_best_2.fit(X_train, Y_train)
RFC_best_pred_2 = RFC_best_2.predict(X_test)

SVM_best = SVC(**SVM_params, random_state=42)
SVM_best.fit(X_train, Y_train)
SVM_best_pred = SVM_best.predict(X_test)

# Matrix class to calculate the accuracy, precision and f1 score given the predictions from each model
def metrics_calc(predictions, true_values):
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions, average='weighted', zero_division=0)  
    f1 = f1_score(true_values, predictions, average='weighted')
    return accuracy, precision, f1

# Prints the accuracy, precision and f1 for each model
accuracy, precision, f1 = metrics_calc(LR_best_pred, Y_test)
print(f"\nLogistic Regression: Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, F1 Score: {round(f1, 4)}")

accuracy, precision, f1 = metrics_calc(RFC_best_pred_1, Y_test)
print(f"\nRandom Forest Classifier 1: Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, F1 Score: {round(f1, 4)}")

accuracy, precision, f1 = metrics_calc(RFC_best_pred_2, Y_test)
print(f"\nRandom Forest Classifier 2: Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, F1 Score: {round(f1, 4)}")

accuracy, precision,f1 =  metrics_calc(SVM_best_pred, Y_test)
print(f"\nSupport Vector Machine Classifier: Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, F1 Score: {round(f1, 4)}")

# Plots the confusion matrix using LR
cm = confusion_matrix(Y_test, LR_best_pred)  
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=RFC.classes_, yticklabels=RFC.classes_) 
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Step 6
estimators = [
    ('lr', LR_best),
    ('rf1', RFC_best_1)
             ]

# Defining the stacked model using StackingClassifier()
stacked_model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
stacked_model.fit(X_train, Y_train)
Y_pred = stacked_model.predict(X_test)

# Prints the accuracy, precision and f1 score for the stacked model
accuracy, precision, f1 = metrics_calc(Y_pred, Y_test)
print(f"\nStacked Model: Accuracy: {round(accuracy, 4)}, Precision: {round(precision, 4)}, F1 Score: {round(f1, 4)}\n")

# Plots the confusion matrix of the stacked model
cm = confusion_matrix(Y_test, Y_pred)  
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=RFC.classes_, yticklabels=RFC.classes_) 
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Stacked Model Confusion Matrix')
plt.show()

# Step 7
# using LR to predict the class based on the given coordinates
LR_best = LogisticRegression(max_iter=10000, **LR_params)
LR_best.fit(X.values, Y)  

filename = 'model.joblib'
joblib.dump(LR_best, filename)

model = joblib.load(filename)
data = [[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]]
pred = model.predict(data)

print(pred)


