import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# IMPORT FEATURE DATA
print('Welcome to the KNN Model Script!')
auth_user = input('Enter the Authorized User ID (1-15): ')
unauth_user = input('Enter the Unauthorized User ID (1-15): ')
User0 = pd.read_csv('processed-feature-data/user' + auth_user +'.csv')
User1 = pd.read_csv('processed-feature-data/user' + unauth_user +'.csv')
final_feature_set = pd.concat([User0, User1], ignore_index=True)


# TRAIN TEST SPLIT
X = final_feature_set
y = final_feature_set['User']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# FIT THE MODEL
knn = KNeighborsClassifier(n_neighbors=10)  # n_neighbors should be adjusted to best value
knn.fit(X_train, y_train)


# TEST MODEL
pred = knn.predict(X_test)


# EVALUATE METRICS
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
