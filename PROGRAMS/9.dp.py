import numpy as np
import pandas as pd
from sklearn import datasets
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

#Splitting the dataset into the Training set and Test  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 2)

#Feature Scaling
"""
Feature scaling is the process of converting the data into a given range. In this case, the standard scalar technique is used.
from sklearn.preprocessing import StandardScaler
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""
Training the Logistic Regression (LR) Classification model on the Training set
Once the dataset is scaled, next, the Logistic Regression (LR) classifier algorithm is used to create a model. 
The hyperparameters such as random_state to 0 respectively.
 The remaining hyperparameters Logistic Regression (LR) are set to default values.
 """
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
#Logistic Regression (LR) classifier model
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
#Display the results (confusion matrix and accuracy)
"""
Here evaluation metrics such as confusion matrix and accuracy are used to evaluate the performance of the model 
built using a decision tree classifier.
"""
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
