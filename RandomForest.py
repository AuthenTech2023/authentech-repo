# Random Forest Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

dtree = DecisionTreeClassifier()

dfXtrain = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')

print('X_train data loaded.')

# Create a smaller subset just to run the data
half = int(len(dfXtrain.index)//100)
dfXtrain_subset = dfXtrain.iloc[0:half]

dfYtrain = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')
print('Y_train data loaded.')

# Create a smaller subset just to run the data
half = int(len(dfYtrain.index)//100)
dfYtrain_subset = dfYtrain.iloc[0:half]

print("Y train subset loaded")

dtree.fit(dfXtrain_subset, dfYtrain_subset)

print('Model fitted.')

# Only trying with user 8 for now
dfXtest = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user8.csv')

# Create more subsets
half_testing = int(len(dfXtest.index)//100)
dfXtest_subset = dfXtest.iloc[0:half_testing]


predictions = dtree.predict(dfXtest_subset)
print("Predictions done.")

print("...")
dfYtest = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user8.csv')
print("Y test values loaded.")

half_Y_test = int(len(dfYtest.index)//100)
dfYtest_subset = dfXtest.iloc[0:half_testing]
print("Y test subset loaded")

print(confusion_matrix(dfYtest_subset,predictions))
print('\n')
#print(classification_report(dfYtest_subset, predictions))