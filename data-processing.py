# Script to process data sets into feature sets

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler

def process_user(user):

    df = pd.read_csv('raw-data/vol' + str(user) + '.csv')

    # X, Y, TOUCH_MINOR, TRACKING_ID have some rows with values of -420 (im assuming an error of sorts)
    # these values throw off feature calculation, so we will drop them (especially in user 4)
    # :this barely helped user 4's issue
    x_mask = df['X'] == -420
    y_mask = df['Y'] == -420
    df = df[~x_mask]
    df = df[~y_mask]

    # DATA CLEANING
    # separate data by finger
    df = df.sort_values(['FINGER','Timestamp'])  # Sort values by finger first, then timestamp


    # todo: handle missing data?
    # either remove rows, remove columns, or fill with data

    # standardization
    # convert string values to numerical values
    df = pd.get_dummies(df,columns=['BTN_TOUCH'],drop_first=True)


    # SEPARATE INTO GESTURES
    # todo: Separate into gestures
    # either by number of data points (suggested) or time period



    # FEATURE EXTRACTION
    df.insert(len(df.columns), "X_Speed", 0)
    df.insert(len(df.columns), "X_Acceleration", 0)
    df.insert(len(df.columns), "Y_Speed", 0)
    df.insert(len(df.columns), "Y_Acceleration", 0)
    df.insert(len(df.columns), "Speed", 0)
    df.insert(len(df.columns), "Acceleration", 0)
    df.insert(len(df.columns), "Jerk", 0)
    df.insert(len(df.columns), "Ang_V", 0)
    df.insert(len(df.columns), "Path_Tangent", 0)

    # I am still unsure if this is how we should be calculating features
    df['X_Speed'] = (df.X - df.X.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Y_Speed'] = (df.Y - df.Y.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Speed'] = np.sqrt((df.X_Speed ** 2) + (df.Y_Speed ** 2))
    df['X_Acceleration'] = (df.X_Speed - df.X_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Y_Acceleration'] = (df.Y_Speed - df.Y_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Acceleration'] = (df.Speed - df.Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Jerk'] = (df.Acceleration - df.Acceleration.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
    df['Path_Tangent'] = np.arctan2((df.Y - df.Y.shift(1)), (df.X - df.X.shift(1)))
    df['Ang_V'] = (df.Path_Tangent - df.Path_Tangent.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))

    # TESTING FEATURES (not permanent changes)
    # df.drop(['Timestamp'], axis=1, inplace=True)
    # df.drop(['X'], axis=1, inplace=True)
    # df.drop(['Y'], axis=1, inplace=True)
    # df.drop(['TRACKING_ID'], axis=1, inplace=True)
    # df.drop(['FINGER'], axis=1, inplace=True)
    # df.drop(['BTN_TOUCH_HELD'], axis=1, inplace=True)
    # df.drop(['BTN_TOUCH_UP'], axis=1, inplace=True)
    # df.drop(['TOUCH_MAJOR'], axis=1, inplace=True)
    # df.drop(['TOUCH_MINOR'], axis=1, inplace=True)

    # Drop missing or NaN values
    df.dropna(inplace=True)

    # scaling
    # todo: features we shouldn't scale? (finger, btn_touch, tracking_id)
    scaler = StandardScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df),columns=df.keys())

    # # normalization
    # for column in df.columns:
    #     df[column] = df[column] / df[column].abs().max()

    # SAVE TO CSV
    df['User'] = user
    df.to_csv('processed-feature-data/user' + str(user) + '.csv')
    print('User ' + str(user) + ' processed')


if __name__ == '__main__':
    for i in range(1,16):
        process_user(i)
