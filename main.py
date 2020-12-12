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
    

"""




import pandas as pd



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