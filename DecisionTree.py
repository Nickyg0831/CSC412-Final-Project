# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:17:40 2020

@author: nicky
"""

from main import data
import matplotlib.pyplot as plt # matplotlib is for drawing graphs
from sklearn.model_selection import train_test_split # split data into training and testing sets
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree

data_no_missing = data.loc[(data['Age'] != 'NaN') & (data['Cabin'] != 'NaN')]
X = data[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].drop('Survived', axis = 1)
y = data['Survived']

feature_cols = list(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# above yields 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
#clf = DecisionTreeClassifier(criterion = "entropy")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print ("\n------------------------------------------------------------\n")
print("Model Accuracy\n")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("Precision ", metrics.precision_score(y_test, y_pred))
print ("Recall ", metrics.recall_score(y_test, y_pred ))
print ("f1 score ", metrics.f1_score(y_test, y_pred ))
print("roc-auc score:", metrics.roc_auc_score(y_test, y_pred))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=feature_cols, 
                   class_names=['0','1'],
                   #class_names=datatext.target_names,
                   filled=True)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print ("\n------------------------------------------------------------\n")
print("Predicted response for test dataset\n")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("Precision ", metrics.precision_score(y_test, y_pred))
print ("Recall ", metrics.recall_score(y_test, y_pred ))
print ("f1 score ", metrics.f1_score(y_test, y_pred ))
print("roc-auc score:", metrics.roc_auc_score(y_test, y_pred))

# The tree has four levels including the root node
# For the tree to come to an accurate decision whether or not
# the patient has heart disease, we need to traverse
# a total of 4 levels in the tree. If we don't include the root, 
# the answer is 3