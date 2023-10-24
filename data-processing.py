# Script to process data sets into feature sets

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import StandardScaler


GESTURE_SIZE = 10

# IMPORT RAW DATA
print('Welcome to the data processing script!')
user = input('Input User to process (1-15): ')
df = pd.read_csv('raw-data/vol' + user + '.csv')


# DATA CLEANING
# separate data by finger
df = df.sort_values(['FINGER','Timestamp'])  # Sort values by finger first, then timestamp

# todo: handle missing data?
    # either remove rows, remove columns, or fill with data

# todo: handle duplicate data?
    # duplicate data increases the chance of overfitting
        # what do we consider duplicate data? what if they just have different timestamps?

# standardization
# convert string values to numerical values
df = pd.get_dummies(df,columns=['BTN_TOUCH'],drop_first=True) # drop first to prevent perfect predictor (overfitting)


# SEPARATE INTO GESTURES
# todo: Separate into gestures
# either by number of data points (suggested) or time period

# we break all the raw data into gestures and extract features of the gesture for added data
# we predict gesture by gesture whether it belongs to user 1 or 2

# todo: options for storing gestures
    # deque/array/dataframe: process the raw data into gestures, then extract features
        # warning from documentation: iterating through pandas objects is slow and often unnecessary
        # performance can be improved in many ways: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        # vectorization: https://stackoverflow.com/questions/1422149/what-is-vectorization

    # don't store: loop through the data, extracting the features from a gesture and storing the feature immediately after gesture is identified
        # does it even matter which one? is this unnecessary optimization?

current_gesture = pd.DataFrame
i = 0  # count variable for rows in gesture

for row in df:
    if i >= GESTURE_SIZE:  # todo: include lifting finger as condition?
        extract_features(current_gesture)
        current_gesture = pd.DataFrame(columns=current_gesture.columns)  # clear data but keep columns
        continue
    current_gesture.append(row)
    i += 1



# for rows in csv
    # while user = 0
        # for i in range gesture size:
            # add to gesture (is the gesture one rows of averages? a full separated collection of rows?
        # add gesture to gesture csv/df
    # while user = 1
        # for i in range gesture size
            # add to gesture
        # add gesture to gesture csv/df



# FEATURE EXTRACTION
# our goal is to output a csv file containing a row of features for each gesture
df.insert(len(df.columns), "X_Speed", 0)
df.insert(len(df.columns), "X_Acceleration", 0)
df.insert(len(df.columns), "Y_Speed", 0)
df.insert(len(df.columns), "Y_Acceleration", 0)
df.insert(len(df.columns), "Speed", 0)
df.insert(len(df.columns), "Acceleration", 0)
df.insert(len(df.columns), "Jerk", 0)
df.insert(len(df.columns), "Ang_V", 0)
df.insert(len(df.columns), "Path_Tangent", 0)
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
            #(I dont think so, we could add all sorts of features for midpoints but we are then essentially splitting one feature into 2)
    # sum of angles: add up all the angles from each data point
    # sharp angles: count how many times the angle change exceeds a given constant
    # time until movement: time taken for a user to move their finger after touching the screen
# how do we determine which features to use?
    # is it possible to test all combinations of features and graph our results? or would that take too long
# should raw data be considered in features?

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

# Drop missing or NaN values
df.dropna(inplace=True)

# scaling
scaler = StandardScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df),columns=df.keys())

# # normalization
# for column in df.columns:
#     df[column] = df[column] / df[column].abs().max()

# SAVE TO CSV
df['User'] = user
df.to_csv('processed-feature-data/user' + user + '.csv')
