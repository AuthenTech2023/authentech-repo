# Random Forest Model
import time
import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import train_test_split

from typing import Any
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

'''
Defining results:

[[TN FP
  FN TP]]
'''

# Going to try fitting the model once and testing it 15 times
# IMPORT TRAIN AND TEST DATA
X_train = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')
y_train = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')

param_grid = {'max_depth':[7, 9, 11, 13], 'max_features':['sqrt','log2'],
              'min_samples_leaf':[2,3,4,5], 'min_samples_split':[7,8,9,10]}
print('param grid set.')

grid = GridSearchCV(RandomForestClassifier(), param_grid, verbose=2)
print('Fitting...')
grid.fit(X_train,y_train.values.ravel())
print("Done.")
print('Grid search complete')

# Found dropping these metrics to be beneficial
X_train = X_train.drop(['Timestamp'], axis=1)

rfc = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=3)

rfc.fit(X_train, y_train.values.ravel())
print('\nModel fitted.\n')

# todo: Add max_features parameter once you figure out how it works

# todo: Also add max_features implementation
def random_forest_model(auth_user, max_depth=7, min_samples_leaf=3):
    X_test = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(auth_user) + '.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(auth_user) + '.csv')

    X_test = X_test.drop(['Timestamp'], axis=1)

    pred = rfc.predict(X_test)

    return confusion_matrix(y_test.values.ravel(), pred)


if __name__ == '__main__':

    # Loop through and run the model for all users

    for i in range(1,16):
        print("User " + str(i) + " results: \n")
        # max_depth=10 giving really really accurate results, and with right sized CM's
        # max_depth=7 seems to be the most optimal??
        # max_depth=6 seems to be giving us pretty good results across the board, with exception to a few users where it is remarkably varying

        # max_features = 'log2' giving me different dimension confusion matrices???
        # i, max_depth=7, min_samples_leaf=3
        print(random_forest_model(i, max_depth=6))

#todo: create a method for finding the max_depth, seemed that this was extremely helpful
