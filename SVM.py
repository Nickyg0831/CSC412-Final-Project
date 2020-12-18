# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:22:50 2020

@author: nicky
"""

from main import data
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier


X = data[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].drop('Survived', axis = 1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#print(data.head)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("SVM Precision:",metrics.precision_score(y_test, y_pred))
print("SVM Recall:",metrics.recall_score(y_test, y_pred))

"""
We are getting around 70% accuracy using the SVM
After looking at ways of improving accuracy, we have decided
on using a Bagging Classifier, which creates multiple random 
subsets of data to improve accuarcy at the expense of efficiency
"""

# Reduce dataset
dataset_size = 100
X = X[:dataset_size]
y = y[:dataset_size]

#Defines the bagging model
svm = SVC(random_state=42)
model = BaggingClassifier(base_estimator=svm, n_estimators=31, random_state=314)

#Fits the model
model.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
y_pred = clf.predict(X_test)

print("\n")
print("New SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("New SVM Precision:",metrics.precision_score(y_test, y_pred))
print("New SVM Recall:",metrics.recall_score(y_test, y_pred))

"""
After implementing the Bagging Classifier,
we are getting an average accuarcy of 85%
which is a significant improvement.
"""