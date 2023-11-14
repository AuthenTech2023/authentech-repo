import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import os


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
    conf_matrix = confusion_matrix(y_test, pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    class_report = classification_report(y_test, pred)
    roc = roc_curve(y_test, pred)

    # print to file (Necessary for confusion-matrix-display.py)
    output_directory = os.path.join("model-outputs", "knn", 'kvalue' + str(k))
    os.makedirs(output_directory, exist_ok=True)
    with open('model-outputs/knn/kvalue' + str(k) + '/user' + str(auth_user) + '_confusion_matrix.txt', mode="w") as f:
        f.write(str(conf_matrix))
    with open('model-outputs/knn/kvalue' + str(k) + '/user' + str(auth_user) + '_classification_report.txt', mode="w") as f:
        f.write(str(class_report))
    with open('model-outputs/knn/kvalue' + str(k) + '/user' + str(auth_user) + '_roc_curve.txt', mode="w") as f:
        f.write(str(roc))

    # print to std
    print(conf_matrix)
    print(class_report)
    print(roc)


    # RETURNS FOR ELBOW METHOD
    return pred, y_test.values.ravel()


if __name__ == '__main__':
    authentic_user = 1
    k = 2
    knn_model(authentic_user, k)

    # multiple users
    # for authentic_user in range(1,16):
    #     print(f'user {authentic_user} start')
    #     knn_model(authentic_user, k)
