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

def SVC_model(auth_user):

    X_train = pd.read_csv('processed-feature-data/training-data2/X_training_data.csv')
    X_test = pd.read_csv('processed-feature-data/testing-data2/X_testing_data_user' + str(auth_user) + '.csv')
    y_train = pd.read_csv('processed-feature-data/training-data2/y_training_data.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data2/y_testing_data_user' + str(auth_user) + '.csv')

    X_train = X_train.drop(['Timestamp'], axis=1)
    X_test = X_test.drop(['Timestamp'], axis=1)

    svc = SVC()

    svc.fit(X_train, y_train)

    predictions = svc.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print('\n')
    # print(classification_report(y_test, predictions))

if __name__ == "__main__":
    '''for i in range(1, 16):
        print("User " + str(i) + " results:")
        SVC_model(i)'''
    SVC_model(15)