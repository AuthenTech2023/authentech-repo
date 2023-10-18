# Sloppy research code to read, preprocess, and evaluate raw mouse dynamics data
# Author: Nyle Siddiqui

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import random
import time
import tensorflow as tf
import pandas as pd

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn import preprocessing, svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, plot_roc_curve
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler
from collections import deque

SEQ_LEN = 10
EXTRACTED_FEATURE_SEQ_LEN = 200
EPOCHS = 1
BATCH_SIZE = 8
NAME = f"SEQ-{SEQ_LEN}-{int(time.time())}"  # For saving model, not in use


# Processing method for 1D-CNN & RNN
def deep_preprocess(df, target):
	pre_seq = []
	seqeuantial = []
	
	df = df.loc[(df["X"].shift() != df["X"]) | (df["Y"].shift() != df["Y"])] # Delete duplicates
	del df['Sex']
	del df['Timestamp']
	del df['Button Pressed']
	del df['Time']
	del df['DistanceX']
	del df['DistanceY']
	
	#binary classification
	df_interest = df[df['Subject ID'] == target]
	df_negative = df[df['Subject ID'] != target]
	
	#create 3-D arrays
	for i in df_interest.values:
		pre_seq.append([n for n in i[:-1]])
		if len(pre_seq) == SEQ_LEN:
			copy = pre_seq.copy()
			seqeuantial.append([copy, 1.0])
			pre_seq.clear()
	half_data = len(seqeuantial) * 2
	pre_seq.clear()
	for j in df_negative.values:
		pre_seq.append([n for n in j[:-1]])
		if len(seqeuantial) < half_data:
			if len(pre_seq) == SEQ_LEN:
				copy = pre_seq.copy()
				seqeuantial.append([copy, 0.0])
				pre_seq.clear()
		else:
			break
	seqeuantial = np.array(seqeuantial)
	random.shuffle(seqeuantial)

	# print((seqeuantial.shape))
	X = []
	y = []
	for seq, target in seqeuantial:
		X.append(seq)
		y.append(target)
	X = np.array(X)
	y = np.array(y)
	return X, y
	
	
#Processing method for taking raw data from 40 users and extracting features for ML models. Feature choice, formulas,
#and even some code taken from this paper: https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/iet-bmt.2018.5126
def extract_forty(df, mode):
	df.rename(columns={"Speed": "X_Speed", "Acceleration": "X_Acceleration"}, inplace=True)
	df.insert(len(df.columns) - 1, "Y_Speed", 0)
	df.insert(len(df.columns) - 1, "Y_Acceleration", 0)
	df.insert(len(df.columns) - 1, "Speed", 0)
	df.insert(len(df.columns) - 1, "Acceleration", 0)
	df.insert(len(df.columns) - 1, "Jerk", 0)
	df.insert(len(df.columns) - 1, "Ang_V", 0)
	df.insert(len(df.columns) - 1, "Path_Tangent", 0)
	df.insert(len(df.columns) - 1, "Direction", 0)
	del df['Sex']  # No need for gender feature anymore
	df = df.loc[(df["X"].shift() != df["X"]) | (df["Y"].shift() != df["Y"])]
	df['X_Speed'] = (df.X - df.X.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
	df['Y_Speed'] = (df.Y - df.Y.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
	df['Speed'] = np.sqrt((df.X_Speed ** 2) + (df.Y_Speed ** 2))
	df['X_Acceleration'] = (df.X_Speed - df.X_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
	df['Y_Acceleration'] = (df.Y_Speed - df.Y_Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
	df['Acceleration'] = (df.Speed - df.Speed.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
	df['Jerk'] = (df.Acceleration - df.Acceleration.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
	df['Path_Tangent'] = np.arctan2((df.Y - df.Y.shift(1)), (df.X - df.X.shift(1)))
	df['Ang_V'] = (df.Path_Tangent - df.Path_Tangent.shift(1)) / (df.Timestamp - df.Timestamp.shift(1))
	# df['Direction'] = df.apply(f, axis=1)
	df = df.fillna(0)
	print(df.head())
	sequential_data = []
	prev_data = deque(maxlen=SEQ_LEN)
	count = 0
	for i in df.values:
		prev_data.append([n for n in i[:-1]])  # Append each row in df to prev_data without 'Subject ID' column, up to 60 rows
		if len(prev_data) == SEQ_LEN:
			temp = np.copy(prev_data)
			for j in range(7, 16):
				temp[0, j] = 0
			mean_x_speed = temp[1:, 7].mean()
			std_x_speed = temp[1:, 7].std()
			min_x_speed = temp[1:, 7].min()
			max_x_speed = temp[1:, 7].max()
			
			mean_x_acc = temp[1:, 8].mean()
			std_x_acc = temp[1:, 8].std()
			min_x_acc = temp[1:, 8].min()
			max_x_acc = temp[1:, 8].max()
			
			mean_y_speed = temp[1:, 9].mean()
			std_y_speed = temp[1:, 9].std()
			min_y_speed = temp[1:, 9].min()
			max_y_speed = temp[1:, 9].max()
			
			mean_y_acc = temp[1:, 10].mean()
			std_y_acc = temp[1:, 10].std()
			min_y_acc = temp[1:, 10].min()
			max_y_acc = temp[1:, 10].max()
			
			mean_speed = temp[1:, 11].mean()
			std_speed = temp[1:, 11].std()
			min_speed = temp[1:, 11].min()
			max_speed = temp[1:, 11].max()
			
			mean_acc = temp[1:, 12].mean()
			std_acc = temp[1:, 12].std()
			min_acc = temp[1:, 12].min()
			max_acc = temp[1:, 12].max()
			
			mean_jerk = temp[1:, 13].mean()
			std_jerk = temp[1:, 13].std()
			min_jerk = temp[1:, 13].min()
			max_jerk = temp[1:, 13].max()
			
			mean_ang = temp[1:, 14].mean()
			std_ang = temp[1:, 14].std()
			min_ang = temp[1:, 14].min()
			max_ang = temp[1:, 14].max()
			
			mean_tan = temp[1:, 15].mean()
			std_tan = temp[1:, 15].std()
			min_tan = temp[1:, 15].min()
			max_tan = temp[1:, 15].max()
			elapsed_time = temp[-1, 0] - temp[0, 0]
			curve_list = list()
			traj_length = 0
			accTimeatBeg = 0
			numCritPoints = 0
			path = list()
			flag = True
			for k in range(1, SEQ_LEN):
				traj_length += np.sqrt((temp[k, 1] - temp[k - 1, 1]) ** 2 + (temp[k, 2] - temp[k - 1, 2]) ** 2)
				path.append(traj_length)
				
				dt = temp[k, 0] - temp[k - 1, 0]
				dv = temp[k, 11] - temp[k - 1, 11]
				if dv > 0 and flag:
					accTimeatBeg += dt
				else:
					flag = False
			for ii in range(1, len(path)):
				dp = path[ii] - path[ii - 1]
				dangle = temp[ii, 15] - temp[ii - 1, 15]
				curv = dangle / dp
				curve_list.append(curv)
				if abs(curv) < .0005:
					numCritPoints += 1
			mean_curve = np.mean(curve_list)
			std_curve = np.std(curve_list)
			min_curve = np.min(curve_list)
			max_curve = np.max(curve_list)
			
			sum_of_angles = np.sum(temp[1:, 15])
			sharp_angles = np.sum(abs(temp[1:, 15]) < .0005)
			
			for jj in [[mean_x_speed, mean_y_speed, mean_speed, mean_acc, mean_jerk, mean_ang, mean_curve,
			            std_x_speed, std_y_speed, std_speed, std_acc, std_ang, std_jerk, std_curve,
			            min_x_speed, min_y_speed, min_speed, min_acc, min_ang, min_jerk, min_curve,
			            max_x_speed, max_y_speed, max_speed, max_acc, max_ang, max_jerk, max_curve,
			            elapsed_time, sum_of_angles, accTimeatBeg, traj_length, numCritPoints, i[-1]]]:
				sequential_data.append(jj)  # Prev_data now contains SEQ_LEN amount of samples and can be appended as one batch of 60 for RNN
		count += 1
		if count % 1000 == 0:
			print(count)
	df = pd.DataFrame(sequential_data,
	                  columns=['mean_x_speed', 'mean_y_speed', 'mean_speed', 'mean_acc', 'mean_jerk', 'mean_ang',
	                           'mean_curve',
	                           'std_x_speed', 'std_y_speed', 'std_speed', 'std_acc', 'std_ang', 'std_jerk', 'std_curve',
	                           'min_x_speed', 'min_y_speed', 'min_speed', 'min_acc', 'min_ang', 'min_jerk', 'min_curve',
	                           'max_x_speed', 'max_y_speed', 'max_speed', 'max_acc', 'max_ang', 'max_jerk', 'max_curve',
	                           'elapsed_time', 'sum_of_angles', 'accTimeatBeg', 'traj_length', 'numCritPoints',
	                           'class'])
	print(df.head())
	df.to_csv(f"master40{mode}_Extracted{SEQ_LEN}.csv")
	
	
#Processing method for ANN
def preprocess(df):
	df = df.sample(frac=1) # Shuffle
	del df['Sex']
	del df['Timestamp']
	del df['Button Pressed']
	# del df['Unnamed: 0']
	del df['Time']
	print(df.head())
	X = np.array(df.iloc[:, :-1])
	y = np.array(df.iloc[:, -1])
	return X, y


#Processing method for ML models
def ml_process(df, target):
	sequential = []
	df_interest = df[df['class'] == target]
	df_negative = df[df['class'] != target]
	df_interest['class'] = 0
	df_negative['class'] = 1
	df_negative = df_negative.sample(frac=1)
	for i in df_interest.values:
		sequential.append([i[:-1], i[-1]])
	half_data = len(sequential)
	for j in df_negative.values:
		if len(sequential) < half_data*2:
			sequential.append([j[:-1], j[-1]])
		else:
			break
	random.shuffle(sequential)
	X = []
	y = []
	for seq, ans in sequential:
		X.append(seq)
		y.append(ans)
	X = np.array(X)
	y = np.array(y)
	return X, y

#Random Forest model generation
def rf(train_X, train_y, test_X, test_y):
	n_estimators = 1600
	max_depth = 30
	min_samples_split=2
	min_samples_leaf=1
	# # Number of trees in random forest
	# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
	# # Number of features to consider at every split
	# max_features = ['auto', 'sqrt']
	# # Maximum number of levels in tree
	# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
	# max_depth.append(None)
	# # Minimum number of samples required to split a node
	# min_samples_split = [2, 5, 10]
	# # Minimum number of samples required at each leaf node
	# min_samples_leaf = [1, 2, 4]
	# # Method of selecting samples for training each tree
	# bootstrap = [True, False]
	# # Create the random grid
	# random_grid = {'n_estimators': n_estimators,
	#                'max_features': max_features,
	#                'max_depth': max_depth,
	#                'min_samples_split': min_samples_split,
	#                'min_samples_leaf': min_samples_leaf,
	#                'bootstrap': bootstrap}
	rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
	                            min_samples_leaf=min_samples_leaf, n_jobs=7, bootstrap=False, random_state=69)
	rf.fit(train_X, train_y)
	predictions = rf.predict(test_X)
	print(classification_report(test_y, predictions, target_names=["0", "1"], digits=4))
	tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
	# class_accuracies = confusion_matrix(test_y, predictions, normalize="true").diagonal()
	# print(f"true acc: {class_accuracies[0]} imposter acc: {class_accuracies[1]}")
	# print(f"ACC: {accuracy_score(real_test_y, predictions)}")
	# print(f"ROC_AUC: {roc_auc_score(real_test_y, prob_predicts[:, 1])}")
	fpr, tpr, threshold = roc_curve(test_y, predictions, pos_label=1)
	fnr = 1 - tpr
	EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	print(EER)
	# print(f"probACC: {accuracy_score(real_test_y, prob_predicts)}")
	# tn, fp, fn, tp = confusion_matrix(test_y, prob_predicts).ravel()
	# print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(f"FPR: {fp / (fp + tn)}")
	print(f"FNR: {fn / (fn + tp)}")
	return rf

#SVM model generation
def sv(train_X, train_y, test_X, test_y):
	rbf = svm.SVC()
	rbf.fit(train_X, train_y)
	predictions = rbf.predict(test_X)
	print(classification_report(test_y, predictions, target_names=["0", "1"]))
	tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
	print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(roc_auc_score(test_y, predictions))
	fpr, tpr, threshold = roc_curve(test_y, predictions, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	print(f"EER: {EER}")
	return rbf
	
#KNN model generatio
def knn(trainx, trainy, testx, testy):
	clf = KNeighborsClassifier(13)
	scaler = MinMaxScaler()
	trainx = scaler.fit_transform(trainx)
	testx = scaler.transform(testx)
	clf.fit(trainx, trainy)
	# print(clf.best_params_)
	predictions = clf.predict(testx)
	prob_predicts = clf.predict_proba(testx)
	print(classification_report(testy, predictions, target_names=["0", "1"]))
	tn, fp, fn, tp = confusion_matrix(testy, predictions).ravel()
	print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(roc_auc_score(testy, predictions))
	fpr, tpr, threshold = roc_curve(testy, prob_predicts[:, 1], pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	prob_predicts = (prob_predicts[:, 1] >= eer_threshold).astype('int')
	print(f"EER: {EER}")
	print(f"prob_predicts: {prob_predicts}")
	return clf
	
	
#RNN model generation
def rnn(trainx, trainy, valx, valy):
	model = Sequential()
	
	model.add(LSTM(128, input_shape=(trainx.shape[1:]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, input_shape=(trainx.shape[1:]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, input_shape=(trainx.shape[1:])))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(1, activation='softmax'))

	opt = tf.keras.optimizers.Adam(learning_rate=5e-3, decay=1e-6)

	model.compile(loss='binary_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])
	tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
	
	filepath = "RNN_Final-{epoch:02d}-{{val_acc:.3f}"
	checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
	model.summary()
	history = model.fit(trainx, trainy,
	                    epochs=EPOCHS,
	                    batch_size=BATCH_SIZE,
	                    validation_data=(valx, valy),
	                    # callbacks=[tensorboard, checkpoint],
	                    verbose=1)
	predictions = model.predict(valx)
	print(classification_report(valy, predictions, target_names=["0", "1"], digits=4))
	tn, fp, fn, tp = confusion_matrix(valy, predictions).ravel()
	fpr, tpr, threshold = roc_curve(valy, predictions, pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	print(EER)
	# print(f"probACC: {accuracy_score(real_test_y, prob_predicts)}")
	# tn, fp, fn, tp = confusion_matrix(test_y, prob_predicts).ravel()
	# print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(f"FPR: {fp / (fp + tp)}")
	print(f"FNR: {fn / (fn + tp)}")
	# model.save('saved_model-real-25.h5')
	# model.save_weights('saved_model-real-25weights.h5')
	# test_df = pd.read_csv("masterTest.csv", skiprows=1,
	#                       names=["Timestamp", "X", "Y", "Button Pressed", "Time", "DistanceX", "DistanceY", "Speed",
	#                              "Acceleration", "Sex", "Subject ID"])
	# test_df.set_index("Timestamp", inplace=False)
	# test_X, test_y = preprocess(test_df)
	# print("Done with test")
	# results = model.evaluate(test_X, test_y, verbose=1)
	
	
#CNN model generation
def cnn(trainx, trainy, valx, valy):
	model = Sequential()
	model.add(layers.Conv1D(filters=40, kernel_size=6, strides=2, activation="elu", input_shape=(trainx.shape[1], trainx.shape[2]), name="thot1"))
	model.add(layers.Conv1D(filters=60, kernel_size=3, strides=1, activation='elu', name="thot2"))
	model.add(layers.GlobalMaxPooling1D(name="thot3"))
	model.add(layers.Flatten())
	model.add(Dense(60, activation='elu', name="thot4"))
	model.add(Dense(1, activation='softmax', name="thot5"))
	opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6)
	model.compile(loss='binary_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])
	model.fit(trainx, trainy,
	                    epochs=EPOCHS,
	                    validation_data=(valx, valy),
	                    # callbacks=[tensorboard, checkpoint],
	                    verbose=1)
	predictions = model.predict(valx)
	print(classification_report(valy, predictions, target_names=["0", "1"], digits=4))
	tn, fp, fn, tp = confusion_matrix(valy, predictions).ravel()
	print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(f"FPR: {fp / (fp + tn)}")
	print(f"FNR: {fn / (fn + tp)}")
	fpr, tpr, threshold = roc_curve(valy, predictions.ravel(), pos_label=1)
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	print(f"EER: {EER}")
	print(roc_auc_score(valy, predictions))

#ANN model generation
def fnn(trainx, trainy, valx, valy):
	model = Sequential()
	print(trainx.shape)
	model.add(Dense(256, input_shape=(trainx.shape[1],), activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(40, activation='softmax'))
	opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-6)
	model.compile(loss='sparse_categorical_crossentropy',
	              optimizer=opt,
	              metrics=['accuracy'])
	tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
	filepath = f"RNN_Final-{EPOCHS:02d}"
	checkpoint = ModelCheckpoint(
		"models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
	model.summary()
	history = model.fit(trainx, trainy,
	                    epochs=EPOCHS,
	                    batch_size=BATCH_SIZE,
	                    validation_data=(valx, valy),
	                    callbacks=[tensorboard, checkpoint],
	                    verbose=1)
	_, acc = model.evaluate()
	predictions = model.predict(valx)
	print(classification_report(valy, predictions,
	                            target_names=["0", "1", '2', '3', "4", "5", '6', '7', "8", "9", '10', '11', "12", "13",
	                                          '14', '15', "16", "17", '18', '19', "20", "21", '22', '23', "24", "25",
	                                          '26', '27',
	                                          "28", "29", '30', '31', "32", "33", '34', '35', "36", "37", '38', '39'],
	                            digits=4))
	tn, fp, fn, tp = confusion_matrix(valy, predictions).ravel()
	# class_accuracies = confusion_matrix(test_y, predictions, normalize="true").diagonal()
	# print("class: 3")
	print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")
	print(f"FPR: {fp / (fp + tn)}")
	print(f"FNR: {fn / (fn + tp)}")
	
	
if __name__ == '__main__':
	np.set_printoptions(threshold=np.inf)
	pd.options.mode.chained_assignment = None
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)
	
	train_df = pd.read_csv(f"masterTrain.csv", header=0)
	del train_df['Unnamed: 0']
	print("Done with train")
	
	val_df = pd.read_csv(f"masterVal.csv", header=0)
	del val_df['Unnamed: 0']
	
	for interest in range(40):
		print(f"Subject: {interest}")
		trainx, trainy = deep_preprocess(train_df, interest)
		valx, valy = deep_preprocess(val_df, interest)
		rnn(trainx, trainy, valx, valy)