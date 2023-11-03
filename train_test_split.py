# This script takes processed feature data files and splits them into our training and testing data
# training_data: 1 csv of all user's training data
# user_x_test_data: csv of testing data for each user

import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sk
from sklearn.model_selection import train_test_split


def load_data(users):
    '''
    Loads processed user data into an array of dataframes
    :param users: list of integer user ids
    :return: array of dataframes
    '''
    dataframes = []
    for user in users:
        df = pd.read_csv('processed-feature-data/user' + str(user) + '.csv')
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        dataframes.append(df)
        print('User ' + str(user) + ' loaded')
    return dataframes


def traintestsplit(user_data):
    '''
    Splits data into one csv of training data and individual csvs of test data
    :param user_data: array of dataframes
    :return: none
    '''
    # todo: y groups may have to be series rather than dataframes
    X_training_data = pd.DataFrame()  # df with all users X training data
    y_training_data = pd.DataFrame()  # df with all users y training data
    X_testing_data = []  # array of X testing data for each user where index+1 is user id
    y_testing_data = []  # array of y testing data for each user where index+1 is user id

    for df in user_data:  # processes each users data individually
        X = df
        y = df['User']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        X_training_data = pd.concat([X_training_data, X_train])
        y_training_data = pd.concat([y_training_data, y_train])
        X_testing_data.append(X_test)  #todo: should we drop 'User' column?
        y_testing_data.append(y_test)

    # Output data to csv files
    X_training_data.to_csv('processed-feature-data/training-data/X_training_data.csv')
    print('X_training_data to csv')
    y_training_data.to_csv('processed-feature-data/training-data/y_training_data.csv')
    print('y_training_data to csv')

    for userid in range(1, len(user_data)+1):
        X_testing_data[userid-1].to_csv('processed-feature-data/testing-data/X_testing_data_user' + str(userid) + '.csv')
        print('X_testing_data User: ' + str(userid) + ' to csv')
        y_testing_data[userid-1].to_csv('processed-feature-data/testing-data/y_testing_data_user' + str(userid) + '.csv')
        print('y_testing_data User: ' + str(userid) + ' to csv')


if __name__ == '__main__':
    users = range(1,16)
    processed_data = load_data(users)
    traintestsplit(processed_data)
    # todo: how do we handle indexes?
