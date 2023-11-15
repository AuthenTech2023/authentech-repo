# SVC Model

import time
import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import train_test_split

from typing import Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

# Look into principal component analysis and grid search

# Also look into passing list of test data


# NEW APPROACH
# Going to try and fit the model once, that is what's taking the longest.
# Just define everything outside the function for now

X_train = pd.read_csv('processed-feature-data/training-data2/X_training_data.csv')
y_train = pd.read_csv('processed-feature-data/training-data2/y_training_data.csv')
X_train = X_train.drop(['Timestamp'], axis=1)
svc = SVC()
svc.fit(X_train, y_train.values.ravel())
print('Model fit')

def SVC_model(auth_user):

    # todo: Make the deep copy, make sure data is in RAM

    X_test = pd.read_csv('processed-feature-data/testing-data2/X_testing_data_user' + str(auth_user) + '.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data2/y_testing_data_user' + str(auth_user) + '.csv')

    X_test = X_test.drop(['Timestamp'], axis=1)

    predictions = svc.predict(X_test)

    return confusion_matrix(y_test, predictions)

    # Don't need
    # print(classification_report(y_test, predictions))

if __name__ == "__main__":
    for i in range(1, 16):
        print("User " + str(i) + " results:")
        print(SVC_model(i))

        cm = SVC_model(i)
        tp = cm[i - 1][i - 1]
        # print(tp)
        fn = sum(cm[i - 1]) - tp
        # print(fn)
        accuracy = (tp) / (fn + tp)
        print("Accuracy: " + str(accuracy))
        print("--------------------------------------------------------")

        # In our implementation, the true positives and false negatives are the same thing