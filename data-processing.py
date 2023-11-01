# Script to process data sets into feature sets

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler


# IMPORT RAW DATA
print('Welcome to the data processing script!')
user = input('Input User to process (1-15): ')
raw_data = pd.read_csv('raw-data/vol' + user + '.csv')


# DATA CLEANING
# separate data by finger
raw_data.sort_values(['FINGER','Timestamp'],inplace=True,ignore_index=True)  # Sort values by finger first, then timestamp

# Drop missing or NaN values (again)
raw_data.dropna(inplace=True,ignore_index=True)

# todo: handle duplicate data?
    # duplicate data increases the chance of overfitting
        # what do we consider duplicate data? what if they just have different timestamps?

# standardization
# convert string values to numerical values
raw_data = pd.get_dummies(raw_data,columns=['BTN_TOUCH'],drop_first=True)  # drop first to prevent perfect predictor (overfitting)


# SEPARATE INTO GESTURES
# todo: Separate into gestures
# we break all the raw data into gestures and extract features of the gesture for added data
# we predict gesture by gesture whether it belongs to user 1 or 2

# todo: options for storing gestures
    # deque/array/dataframe: process the raw data into gestures, then extract features
        # warning from documentation: iterating through pandas objects is slow and often unnecessary
        # performance can be improved in many ways: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        # vectorization: https://stackoverflow.com/questions/1422149/what-is-vectorization

    # don't store: loop through the data, extracting the features from a gesture and storing the feature immediately after gesture is identified
        # does it even matter which one? is this unnecessary optimization?


# FEATURE EXTRACTION
# our goal is to output a csv file containing a row of features for each gesture

def extract_features(gesture):
    gesture.insert(len(gesture.columns), "X_Speed", 0)
    gesture.insert(len(gesture.columns), "X_Acceleration", 0)
    gesture.insert(len(gesture.columns), "Y_Speed", 0)
    gesture.insert(len(gesture.columns), "Y_Acceleration", 0)
    gesture.insert(len(gesture.columns), "Speed", 0)
    gesture.insert(len(gesture.columns), "Acceleration", 0)
    gesture.insert(len(gesture.columns), "Jerk", 0)

    gesture['X_Speed'] = (gesture.X - gesture.X.shift(len(gesture) - 1)) / (
                gesture.Timestamp - gesture.Timestamp.shift(len(gesture) - 1))
    gesture['Y_Speed'] = (gesture.Y - gesture.Y.shift(len(gesture) - 1)) / (
                gesture.Timestamp - gesture.Timestamp.shift(len(gesture) - 1))
    gesture['Speed'] = ((gesture.X_Speed ** 2) + (gesture.Y_Speed ** 2)) ** 0.5
    gesture['X_Acceleration'] = (gesture.X_Speed - gesture.X_Speed.shift(len(gesture) - 1)) / (
            gesture.Timestamp - gesture.Timestamp.shift(len(gesture) - 1))
    gesture['Y_Acceleration'] = (gesture.Y_Speed - gesture.Y_Speed.shift(len(gesture) - 1)) / (
            gesture.Timestamp - gesture.Timestamp.shift(len(gesture) - 1))
    gesture['Acceleration'] = (gesture.Speed - gesture.Speed.shift(len(gesture) - 1)) / (
            gesture.Timestamp - gesture.Timestamp.shift(len(gesture) - 1))
    gesture['Jerk'] = (gesture.Acceleration - gesture.Acceleration.shift(len(gesture) - 1)) / (
            gesture.Timestamp - gesture.Timestamp.shift(len(gesture) - 1))

    return gesture
    # possible additional features:
    # todo: does additional features increase/decrease over/underfitting?
    # size/range of gesture: the area of screen touched
    # distance travelled
    # distance end to end
    # average direction: direction end to end
    # curvature
    # type should each gesture have a single type? should we divide gestures by type? should we include time off screen as a type?
    # smoothness: how smooth the motion is vs jittery/jagged
    # elapsed time
    # additionally we should include min, max, avg, std for features when we can
    # x and y start and end points
    # is it valuable to break this up and include midpoints
    # (I dont think so, we could add all sorts of features for midpoints but we are then essentially splitting one feature into 2)
    # sum of angles: add up all the angles from each data point
    # sharp angles: count how many times the angle change exceeds a given constant
    # time until movement: time taken for a user to move their finger after touching the screen
    # how do we determine which features to use?
    # is it possible to test all combinations of features and graph our results? or would that take too long
    # should raw data be considered in features?

    # ** MANUALLY SELECT FEATURE SETS TO TEST

    # I am still unsure if this is how we should be calculating features



current_gesture = pd.DataFrame(columns=raw_data.columns)

GESTURE_LENGTH = 10
index = 0
DATA_LENGTH = len(raw_data)
print(DATA_LENGTH)

features = pd.DataFrame(columns=raw_data.columns)
features.insert(len(features.columns), "X_Speed", 0)
features.insert(len(features.columns), "X_Acceleration", 0)
features.insert(len(features.columns), "Y_Speed", 0)
features.insert(len(features.columns), "Y_Acceleration", 0)
features.insert(len(features.columns), "Speed", 0)
features.insert(len(features.columns), "Acceleration", 0)
features.insert(len(features.columns), "Jerk", 0)
features.head()

while index + 10 < DATA_LENGTH:
    index += 10
    if index % 1000 == 0:
        print(index)

    current_gesture = raw_data[raw_data.index < index]  # put first 10 rows in current_gesture
    raw_data.drop(raw_data.index[:10], inplace=True)  # drop first 10 rows

    print(extract_features(current_gesture))


# Drop missing or NaN values (again)
features = features.dropna(inplace=True)

# scaling
scaler = StandardScaler()
scaler.fit(features)
features = pd.DataFrame(scaler.transform(features),columns=features.keys())

# # normalization
# for column in df.columns:
#     df[column] = df[column] / df[column].abs().max()

# SAVE TO CSV
features['User'] = user
features.to_csv('processed-feature-data/user' + user + '.csv')
