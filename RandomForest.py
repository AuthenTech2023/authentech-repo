# Random Forest Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


'''

# Found that between max depth of 6 and 7 is where we lose the 100% accuracy
dtree = DecisionTreeClassifier(max_depth=6)

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

def decision_tree_model(auth_user, max_depth=7):

    X_train = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')
    X_test = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(auth_user) + '.csv')
    y_train = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(auth_user) + '.csv')

    dtree = DecisionTreeClassifier(max_depth=max_depth)
    dtree.fit(X_train, y_train)

    pred = dtree.predict(X_test.values)

    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    return pred, y_test


if __name__ == '__main__':
    authentic_user = 8
    # Max depth of 7 makes sense for the results
    max_depth = 7
    decision_tree_model(authentic_user, max_depth)