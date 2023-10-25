import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# MEETING NOTES

# STEP 1: VALIDATION
# TRAIN WITH 70% ONE USERS DATA
# TEST WITH 30% THE SAME USERS DATA
# RESULT SHOULD BE 90-100%

# STEP 2: TRAINING MODEL
# TRAIN THE MODEL WITH 70% OF ALL USERS DATA, INCLUDING USER IDS

# STEP 3: TEST MODEL
# TEST THE MODEL INDIVIDUALLY FOR EACH USER
    # GIVE THE MODEL THE REMAINING 30% OF A USERS DATA
# IDEALLY THERE WOULD BE 100% TRUE POSITIVES, NO TRUE NEGATIVES
    # BECAUSE YOU ARE ONLY FEEDING IT DATA FROM THE "AUTH USER"



def knn_model(au, uau, k=4):
    User0 = pd.read_csv('processed-feature-data/user' + au +'.csv')
    User1 = pd.read_csv('processed-feature-data/user' + uau +'.csv')
    final_feature_set = pd.concat([User0, User1], ignore_index=True)


    # TRAIN TEST SPLIT
    X = final_feature_set
    y = final_feature_set['User']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


    # FIT THE MODEL
    knn = KNeighborsClassifier(n_neighbors=k)  # n_neighbors should be adjusted to best value
    knn.fit(X_train, y_train)


    # TEST MODEL
    pred = knn.predict(X_test)


    # EVALUATE METRICS
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

    # RETURNS FOR ELBOW METHOD
    return pred, y_test


if __name__ == '__main__':
    # IMPORT FEATURE DATA
    print('Welcome to the KNN Model Script!')
    auth_user = input('Enter the Authorized User ID (1-15): ')
    unauth_user = input('Enter the Unauthorized User ID (1-15): ')
    knn_model(auth_user, unauth_user)
