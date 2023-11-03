# Random Forest Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Found that between max depth of 6 and 7 is where we lose the 100% accuracy
dtree = DecisionTreeClassifier(max_depth=7)

dfXtrain = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')

print('X train data loaded.')

dfYtrain = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')
print('Y_train data loaded.')

dtree.fit(dfXtrain, dfYtrain)
print(dtree.max_depth)

print('Model fitted.')

# Only trying with user 11 for now
dfXtest = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user5.csv')

predictions = dtree.predict(dfXtest)
print("Number of predictions: " + str(len(predictions)))

print("...")
dfYtest = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user5.csv')
print("Y test values loaded.")



print(confusion_matrix(dfYtest,predictions))
print('\n')
print(classification_report(dfYtest, predictions))

# Getting all false negatives for results?? Not sure why that is

'''
# Moving on to random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)
# How do we determine what the best n_estimators value is?

rfc.fit(dfXtrain,dfYtrain)
"RFC Fitted."

pred = rfc.predict(dfXtest)
print(confusion_matrix(dfYtest,pred))
print('\n')
print(classification_report(dfYtest,pred))
'''


'''
Defining results:

[[TN FP
  FN TP]]


'''

#def DecisionTree(auth_user, max_depth):

# todo: get rid of all the subset nonsense
# todo: make more object oriented