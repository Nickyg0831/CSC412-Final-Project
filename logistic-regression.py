# -*- coding: utf-8 -*-
"""
Logistic Regression Model
Classification
"""

from main import data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.linear_model import LogisticRegression # Logistic Regression

# dropping features that intuitively don't seem like they are important
data.drop(['Name','Ticket', 'SibSp', 'Parch', 'PassengerId'], 1, inplace=True)

# converting 'Title' categorical column to booleans
X_encoded = pd.get_dummies(data, columns=['Title'])

data = X_encoded

print("data.head()\n", data.head())

X = data.drop("Survived",axis=1)
y = data["Survived"]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("data.head():\n", data.head(),
      "\n\nX_train.shape: ", X_train.shape,
      "\nX_test.shape: ", X_test.shape,
      "\ny_train.shape: ", y_train.shape,
      "\ny_test.shape: ", y_test.shape)

"""
Here we do a 70/30 split between training and test data for the logistic regression model
Below we create the model and apply the training
And from there we test our algorithm against test data results

We were able to achieve a Logistic Regression accuracy of: 83.21%
With a Mean Cross Validation Accuracy score of: 83.05
"""
model = LogisticRegression()
model.fit(X_train,y_train)
prediction_lr=model.predict(X_test)

print('\n')
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Logistic Regression is',round(accuracy_score(prediction_lr,y_test)*100,2))
# 83.21%

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_lr=cross_val_score(model,X,y,cv=10,scoring='accuracy')
print('The Mean Cross Validated Score for Logistic Regression is:',round(result_lr.mean()*100,2))
# 83.05

# range of scores
scores = pd.Series(result_lr)
print("\nMCV min: ", round(scores.min(),2),
      "\nMCV max: ", round(scores.max(),2))
#MCV min:  0.76 
#MCV max:  0.87

print('-------------------------------------------------------------------')

y_pred = cross_val_predict(model,X,y,cv=10)
sns.heatmap(confusion_matrix(y,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)