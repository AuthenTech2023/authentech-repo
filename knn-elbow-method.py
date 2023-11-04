import numpy as np
import pandas as pd
import sklearn as sk
from KNN import knn_model
from matplotlib import figure
import matplotlib as plt


def elbow_method(auth_user, test_range):

    error_rate = []  # holds all error rates for each k value

    for k in test_range:
        pred, y_test = knn_model(auth_user, k)  # knn_model returns pred: predicted class and y_test: actual class
        avg_error_rate = np.mean(pred != y_test)
        print(str(k) + ': ' + str(avg_error_rate))
        error_rate.append(avg_error_rate)

    # Save output to file
    with open('elbow-output.txt', 'w') as f:
        for item in error_rate:
            f.write(str(item))

    # create plot of results
    # TODO: fix this, it should create a plot of error rates
    #plt.figure(figsize=(10,6))

    #plt.plot(range(1,40), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)

    best_k = error_rate.index(min(error_rate)) + 1
    print('Best k = ' + str(best_k) + ' for user ' + str(auth_user))  # print k value with the lowest error rate
    return best_k


if __name__ == '__main__':

    testing_range = range(1, 11)

    # Test for 1 user
    # authentic_user = 1
    # elbow_method(authentic_user, testing_range)

    # Test for all users
    error_values = []
    for user in range(1, 16):
        error_values.append(elbow_method(user, testing_range))

    best_k = max(set(error_values), key=error_values.count)  # calculates mode
    print('Most common best k: ' + str(best_k))
