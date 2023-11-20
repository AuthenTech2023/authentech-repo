# SVC Model

import time
# import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

import train_test_split

from typing import Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.svm import SVC
import os

# Look into principal component analysis and grid search

# Also look into passing list of test data


# NEW APPROACH
# Going to try and fit the model once, that is what's taking the longest.
# Just define everything outside the function for now

X_train = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')
y_train = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')
X_train = X_train.drop(['Timestamp'], axis=1)
svc = SVC()
svc.fit(X_train, y_train.values.ravel())
print('Model fit')

def SVC_model(auth_user):

    # todo: Make the deep copy, make sure data is in RAM

    X_test = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(auth_user) + '.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(auth_user) + '.csv')

    X_test = X_test.drop(['Timestamp'], axis=1)

    pred = svc.predict(X_test)

    # EVALUATE METRICS
    conf_matrix = confusion_matrix(y_test, pred, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    class_report = classification_report(y_test, pred)
    pred = list(pred)

    # print to file (Necessary for confusion-matrix-display.py)
    output_directory = os.path.join("model-outputs", "svc")
    os.makedirs(output_directory, exist_ok=True)
    with open('model-outputs/svc/' + '/user' + str(auth_user) + '_confusion_matrix.txt', mode="w") as f:
        f.write(str(conf_matrix))
    with open('model-outputs/svc/' + '/user' + str(auth_user) + '_classification_report.txt',mode="w") as f:
        f.write(str(class_report))
    with open('model-outputs/svc/' + '/user' + str(auth_user) + '_pred.txt',mode="w") as f:
        f.write(str(pred))

    return conf_matrix

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
