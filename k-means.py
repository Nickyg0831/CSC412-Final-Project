# -*- coding: utf-8 -*-
"""
K-Means Clustering
Using 2 groups (survivors/non-survivors)
k-means is not a classification tool

**NOTE ON ACCURACY**
The group that Survived can be either represented as 0 or a 1!
This is due to the degree of randomness in the algorithm.
Therefore, sometimes the accuracy may appear to be low.
In the accuracy test, we assume survived corresponds to 1.
So, if in the algorithm survived is actually represented as a 0,
take (100%-predictedAccuracy).
Or, you can just keep re-running the algorithm and see if the results are consistent.
"""
from main import data
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

# keeping an original copy of data so we can reuse it
og_data = data.copy()

# Columns with type of object need to be dropped
# We want to convert all datatypes to a numerical value that we can run k-means on
data.drop(['Title','Name','Ticket'], 1, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

# convert any non-numerical data into something we can use
data = handle_non_numerical_data(data)

# X contains the variables which we will draw our prediction from
X = np.array(data.drop(['Survived'], 1).astype(float))

# y contains the variable we want to predict (survival)
y = np.array(data['Survived'])

# clustering with 2 groups
clf = KMeans(n_clusters=2)
clf.fit(X)

"""
The two clusters were established
Based on the data, objects are assigned to a cluster
0 or 1 can arbritrarily represent survivability. It is random.
We can test the accuracy by comparing the predicted survivability of 
an individual to the actual survivability
"""

# correct is a counter that increments if predicted survivability = actual
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("K-Means Classification Accuracy Before Pre-Processing: ", correct/len(X))
#0.5084175084175084

"""
With this current setup, our accuracy is estimated to be around 50%.
This low accuracy can be attributed to outliers.
It can also be attributed to anamolies during data pruning.
We can try to improve our accuracy by applying Pre-Processing.
Pre-Processing will normalize the data and eliminate any outliers.
It will also aim to put the data in a range from -1 to +1.
"""

X = preprocessing.scale(X)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("K-Means Classification Accuracy After Pre-Processing: ", correct/len(X))
#0.6980920314253648
#0.30190796857463525

"""
After pre-processing, our accuracy is estimated to be around 70%.
Note: The group that Survived can be either 0 or a 1!
This is due to the degree of randomness in the algorithm.
Sometimes the accuracy appears to be low,
like around 30% but on sub-sequent trials, we get 70%.
Thus, with the current setup, the model is 70% accurate.

What if we try dropping some columns? How will that change the accuracy?
"""

# How does family affect accuracy? Lets try to ignore these variables.
# Dropping SibSp, Parch, and Family_Size columns

data.drop(['SibSp','Parch','Family_Size'], 1, inplace=True)
X = np.array(data.drop(['Survived'], 1).astype(float))
y = np.array(data['Survived'])
X = preprocessing.scale(X)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("K-Means Classification Accuracy After Dropping Family: ", correct/len(X))
#0.6980920314253648

"""
After dropping the family related variables, there was no change in accuracy.
This is interesting, because one would think that family size would be a
significant factor in ones survivability. 
A smaller family would mean less people to worry about.

Let's continue dropping columns. Now we will drop the Embarked columns.
"""

data.drop(['Embarked_C', 'Embarked_Q', 'Embarked_S'], 1, inplace=True)
X = np.array(data.drop(['Survived'], 1).astype(float))
y = np.array(data['Survived'])
X = preprocessing.scale(X)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("K-Means Classification Accuracy After Dropping Embarked Columns: ", correct/len(X))
#0.7037037037037037

"""
After dropping the Embarked columns, there was a minor change in accuracy of about 1%.
I wonder if there is a correlation between Fare and Survived.
My guess is that a higher fare meant a higher chance at survival.
This is mainly because the wealthy were most likely prioritized.
I will now drop the Sex, Age, Cabin, Pclass columns.
This will leave: PassengerId, Survived, and Fare in the dataframe
"""

data.drop(['Sex', 'Age', 'Cabin', 'Pclass'], 1, inplace=True)
X = np.array(data.drop(['Survived'], 1).astype(float))
y = np.array(data['Survived'])
X = preprocessing.scale(X)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("K-Means Classification Accuracy After Dropping Sex, Age, Cabin, Pclass: ", correct/len(X))
#0.5072951739618407

"""
After dropping everything but the PassengerId, and Fare columns
We surprisingly only have an accuracy of about 50%.
I suppose this means that for the algorithm, and for the people interpreting
the categorization of survivability based on fare alone is not accurate enough.

Does dropping PassangerId have any affect?
"""

data.drop(['PassengerId'], 1, inplace=True)
X = np.array(data.drop(['Survived'], 1).astype(float))
y = np.array(data['Survived'])
X = preprocessing.scale(X)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("K-Means Classification Accuracy for only Fare: ", correct/len(X))
#0.6442199775533108

"""
It appears that dropping PassengerId has a significant affect on the result.
This seemed to improve the K-Means Classification Accuracy Before Pre-Processing
by roughly 14 percent, however for the other accuracies there was a minor difference.
This is very unusual, because it is not like order should effect the accuracy in any way.

What if we were to take the original dataframe, and remove everything but age, sex, and survived
"""

og_data.drop(['PassengerId', 'Pclass', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title', 'Family_Size'], 1, inplace=True)
X = np.array(og_data.drop(['Survived'], 1).astype(float))
y = np.array(og_data['Survived'])
X = preprocessing.scale(X)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
        
print("K-Means Classification Accuracy for only Age, Sex: ", correct/len(X))
#0.7867564534231201

"""
Interesting! It appears that by considering only Age and Sex, the machine was able to group the data
into clusters wth approximately 78% accuracy. 
From this (and based on factual information), we can assume that
if an individual was a woman or a child, they were evacuated first and had higher probability of survival.

Now let us only consider sex.
"""

og_data.drop(['Age'], 1, inplace=True)
X = np.array(og_data.drop(['Survived'], 1).astype(float))
y = np.array(og_data['Survived'])
X = preprocessing.scale(X)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
        
print("K-Means Classification Accuracy for only Sex: ", correct/len(X))
#0.7867564534231201

"""
The result after only considering sex is fairly similar to the accuracy when Age and Sex parameters were considered.
Roughly 78% accuracy.
"""