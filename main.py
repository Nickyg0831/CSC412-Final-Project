# -*- coding: utf-8 -*-
"""

Created on Thurs Dec 10 2020

@authors:
    Ahmed, Yoseph
    Giasi, Nicholas
    Fernando, Nattandige

Project:
    Using a dataset, we will predict whether a passenger of the Titanic 
    will be able to survive or not
    
Github repo:
    https://github.com/Nickyg0831/CSC412-Final-Project

"""




import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')

pd.set_option('display.max_columns', None)

print(data.head(), "\n")

data.columns = ['PassengerId',
                'Survived',
                'Pclass',
                'Name',
                'Sex',
                'Age',
                'SibSp',
                'Parch',
                'Ticket',
                'Fare',
                'Cabin',
                'Embarked']

print(data.dtypes, "\n")

#Encodes "Embarked" column to to boolean
data_no_missing = data.loc[(data['Age'] != 'NaN') & (data['Cabin'] != 'NaN')]
X = data_no_missing.drop('Survived', axis=1).copy()
X_encoded = pd.get_dummies(X, columns=['Embarked'])
X_encoded.head()
print("x_encoded.head():", X_encoded.head())
                                      

print("Sex column:\n", data['Sex'])        
print("Cabin Column:\n", data['Cabin'])     

#Makes "Male" = 1 and "Female" = 0 and changes to datatype bool
data['Sex'] = data['Sex'].replace({'male': 1, 'female': 0})
data['Sex'] = data['Sex'].astype('bool')
print("Sex column:\n", data['Sex'])        

#Fills all "NaN" with 0 and changes all other values to 1 and makes datatype bool
data['Cabin'] = data['Cabin'].fillna(0)
data['Cabin'] = data['Cabin'].astype('bool')
print("Cabin Column:\n", data['Cabin'])     

#Fills all "Nan" with 100
data['Age'] = data['Age'].fillna(100)
print("Age column:\n", data['Age'])

