import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def knn_model(auth_user, k=4):

    # IMPORT TRAIN AND TEST DATA
    X_train = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')
    X_test = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(auth_user) + '.csv')
    y_train = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(auth_user) + '.csv')


    # FIT THE MODEL
    knn = KNeighborsClassifier(n_neighbors=k)  # n_neighbors should be adjusted to best value
    knn.fit(X_train.values, y_train.values.ravel())

    # TEST MODEL
    pred = knn.predict(X_test.values)

    # EVALUATE METRICS
    # print(confusion_matrix(y_test, pred))
    # print(classification_report(y_test, pred))

    # RETURNS FOR ELBOW METHOD
    return pred, y_test.values.ravel()


if __name__ == '__main__':
    authentic_user = 1
    k = 4
    knn_model(authentic_user, k)
