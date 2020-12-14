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


# Encodes "Embarked" column to to boolean
data_no_missing = data.loc[(data['Age'] != 'NaN') & (data['Cabin'] != 'NaN')]
X = data_no_missing.drop('Survived', axis=1).copy()
X_encoded = pd.get_dummies(X, columns=['Embarked'])
                                          

# Makes "Male" = 1 and "Female" = 0 and changes to datatype bool
data['Sex'] = data['Sex'].replace({'male': 1, 'female': 0})
data['Sex'] = data['Sex'].astype('bool')       

# Fills all "NaN" with 0 and changes all other values to 1 and makes datatype bool
data['Cabin'] = data['Cabin'].fillna(0)
data['Cabin'] = data['Cabin'].astype('bool')    

# Fills all "Nan" with 100
data['Age'] = data['Age'].fillna(100)

#print("data.head()\n", data.head(), "\n")
#print("data.dtypes\n", data.dtypes, "\n")

""" 
Here, we accomplished three things:
We have fixed any data fields with missing data,
We have created a boolean for sex and for cabin.

The 'Name' column currently has no function.
We can iterate through the values and pull the title of the person.
We can then recombine all titles into four general sub-categories.
This will result in a new column called 'Title'.

We will also create a new column called 'Family_Size' for family size.
From it we can obtain some useful information.

This will be done following a tutorial by TriangleInequality:
    https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
"""

# All possible titles in 'Name' column
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

# This function searches for sub-strings inside of a string
# This is useful because the names follow a general pattern: Braund, Mr. Owen Harris
def substrings_in_string(String, subStrings):
    for subString in subStrings:
        if str.find(String, subString) != -1:
            return subString
    return np.nan

# Here we map the title found in the name to a new column called Title
data['Title']=data['Name'].map(lambda x: substrings_in_string(x, title_list))
 
# Replacing all titles with Mr, Mrs, Miss, Master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
# Call the replace_titles function on each row of data and change the title
data['Title']=data.apply(replace_titles, axis=1)

# Creating new Family_Size column
data['Family_Size']=data['SibSp']+data['Parch']