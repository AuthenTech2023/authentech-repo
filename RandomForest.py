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

def find_best_classifier(model: str, classifier: Any, param_grid: dict[str, Any], X_train: np.ndarray,
                         y_train: np.ndarray) -> Any:
    """

    :param model: The name of the model, only for debug messages
    :param classifier: The classifier to base the GridSearch off of
    :param param_grid: A dictionary of hyperparameters and their possible values
    :param X_train: Feature training data
    :param y_train: Feature classifier data
    :return: The classifier with the best set of hyperparameters found. Those hyperparameters are also printed out.
    """


    # grid search to find the best possible hyperparameters
    '''
    print(f"> Starting GridSearch for {model}")
    gridsearch = GridSearchCV(classifier, param_grid, cv=constants.CROSS_VALIDATION_STEPS, n_jobs=constants.N_JOBS,
                              verbose=constants.VERBOSE)

    start_time = time.time()
    gridsearch.fit(X_train, y_train)
    print(f"> {model} GridSearch time: {round(time.time() - start_time, constants.NUM_ROUNDING)}s")

    best_params = gridsearch.best_params_
    print(f"> Best parameters for {model} were: {best_params}")
    return gridsearch.best_estimator_
    """
    '''

    '''
    Results:
    
    max_depths:
10
10
10
10
...
10

avg max depth: 10

max_features:
sqrt
log2
log2
sqrt
log2
log2
log2
log2
log2
log2
sqrt
sqrt
log2
log2
sqrt

avg max features: ...
mode max features: log2

min samples:
2
3
2
3
5
5
2
5
5
3
2
5
2
2
3

avg min samples: ~3.267
mode min samples: 2

min samples split:
9
5
10
9
5
7
7
10
5
5
9
10
7
10
10

avg min samples split:~7.867
avg min samples mode: 10
'''
    '''


# Found that between max depth of 6 and 7 is where we lose the 100% accuracy
dtree = DecisionTreeClassifier(max_depth=10)

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

# todo: Add max_features parameter once you figure out how it works

def decision_tree_model(auth_user, max_depth=7, min_samples_leaf=3):

    X_train = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')
    X_test = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(auth_user) + '.csv')
    y_train = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(auth_user) + '.csv')

    # Found that dropping the timestamp from the data helps with the results
    X_train = X_train.drop(['Timestamp'], axis=1)
    X_test = X_test.drop(['Timestamp'], axis=1)

    # Run a grid search
    # From last semester's code
    dtree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    param_grid = {'max_depth': [5, 7, 9, 10], 'min_samples_leaf': [2, 3, 5], 'min_samples_split': [5, 7, 9, 10],
                  'max_features': ["auto", "sqrt", "log2"]}
    #classifier = find_best_classifier("Decision Tree", dtree, param_grid, X_train, y_train)

    dtree.fit(X_train, y_train)

    pred = dtree.predict(X_test)

    print(confusion_matrix(y_test, pred))
    #print(classification_report(y_test, pred))
    print('\n')

    return pred, y_test

# todo: Also add max_features implementation
def random_forest_model(auth_user, max_depth=7, min_samples_leaf=3):

    X_train = pd.read_csv('processed-feature-data/training-data/X_training_data.csv')
    X_test = pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(auth_user) + '.csv')
    y_train = pd.read_csv('processed-feature-data/training-data/y_training_data.csv')
    y_test = pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(auth_user) + '.csv')

    X_train = X_train.drop(['Timestamp'], axis=1)
    X_test = X_test.drop(['Timestamp'], axis=1)

    # Run a grid search
    # From last semester's code
    rfc = RandomForestClassifier(n_estimators=300, max_depth=max_depth, min_samples_leaf=3)
    param_grid = {'max_depth': [5, 7, 9, 10], 'min_samples_leaf': [2, 3, 5], 'min_samples_split': [5, 7, 9, 10],
                  'max_features': ["auto", "sqrt", "log2"]}
    # classifier = find_best_classifier("Decision Tree", dtree, param_grid, X_train, y_train)

    rfc.fit(X_train, y_train)

    pred = rfc.predict(X_test)

    print(confusion_matrix(y_test, pred))
    # print(classification_report(y_test, pred))
    print('\n')

    return pred, y_test


if __name__ == '__main__':

    # Best test so far
    '''
    authentic_user = 8
    # Max depth of 7 makes sense for the results
    max_depth = 7
    decision_tree_model(authentic_user, max_depth)
    '''
    # For some reason, for user 9 we are getting different dimensions of the confusion matrix
    # why???
    #decision_tree_model(9)


    # Loop through and run the model for all users

    for i in range(1,16):
        print("User " + str(i) + " results: \n")
        # max_depth=10 giving really really accurate results, and with right sized CM's
        # max_depth=7 seems to be the most optimal??
        # max_depth=6 seems to be giving us pretty good results across the board, with exception to a few users where it is remarkably varying

        # max_features = 'log2' giving me different dimension confusion matrices???
        decision_tree_model(i, max_depth=7, min_samples_leaf=3)

        # RANDOM FOREST
        #random_forest_model(i, max_depth=7, min_samples_leaf=3)


#todo: create a method for finding the max_depth, seemed that this was extremely helpful
