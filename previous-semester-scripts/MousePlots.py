import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
SEQ_LEN = 60
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates


def preprocess(df):
	# normalize data
	for col in df.columns:
		if col != "Subject ID":
			df[col] = preprocessing.normalize([i[:-1] for i in df.values], axis=1)
	df = df.values
	# append prev data? why?
	sequential_data = []
	prev_data = deque(maxlen=SEQ_LEN)
	for i in df:
		prev_data.append([n for n in i[:-1]])
		if len(prev_data) == SEQ_LEN:
			sequential_data.append([np.array(prev_data), i[-1]])
	random.shuffle(sequential_data)
	X = []
	y = []
	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)
	return np.array(X), np.array(y)


if __name__ == '__main__':
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)
	# train_df = pd.read_csv("masterTrain.csv", skiprows=1, names =["Timestamp", "X", "Y", "Button Pressed", "Time", "DistanceX", "DistanceY", "Speed", "Acceleration", "Sex", "Subject ID"])
	# val_df = pd.read_csv("masterTrain.csv", skiprows=1, names =["Timestamp", "X", "Y", "Button Pressed", "Time", "DistanceX", "DistanceY", "Speed", "Acceleration", "Sex", "Subject ID"])
	test_df = pd.read_csv("Collected/Subject3.csv", skiprows=1, names =["Timestamp", "X", "Y", "Button Pressed", "Time", "DistanceX", "DistanceY", "Speed", "Acceleration", "Sex", "Subject ID"], usecols=['Timestamp', 'X', 'Y'])
	test_df2 = pd.read_csv("Collected/Subject4.csv", skiprows=1,
	                      names=["Timestamp", "X", "Y", "Button Pressed", "Time", "DistanceX", "DistanceY", "Speed",
	                             "Acceleration", "Sex", "Subject ID"], usecols=['Timestamp', 'X', 'Y'])
	test_df3 = pd.read_csv("Collected/Subject5.csv", skiprows=1, names =["Timestamp", "X", "Y", "Button Pressed", "Time", "DistanceX", "DistanceY", "Speed", "Acceleration", "Sex", "Subject ID"], usecols=['Timestamp', 'X', 'Y'])

	# train_df.set_index("Timestamp", inplace=True)
	# val_df.set_index("Timestamp", inplace=True)
	test_df.set_index("Timestamp", inplace=True)
	test_df2.set_index("Timestamp", inplace=True)
	test_df3.set_index("Timestamp", inplace=True)
	plt.rcParams.update({'font.size' : 12})

	test_df = test_df.to_numpy()
	test_df2 = test_df2.to_numpy()
	test_df3 = test_df3.to_numpy()
	x, y = np.hsplit(test_df, 2)
	
	plt.plot(x, y, color='red')
	plt.xlabel('Screen x-coordinates')
	plt.ylabel('Screen y-coordinates')
	plt.title('User 3')
	plt.plot(x, y, color='k')
	plt.show()
	
	x, y = np.hsplit(test_df2, 2)
	plt.xlabel('Screen x-coordinates')
	plt.ylabel('Screen y-coordinates')
	plt.title('User 4')
	plt.plot(x, y, color='m')
	plt.show()
	
	x, y = np.hsplit(test_df3, 2)
	plt.xlabel('Screen x-coordinates')
	plt.ylabel('Screen y-coordinates')
	plt.title('User 5')
	plt.plot(x, y)
	plt.plot(x, y, color='olive')
	plt.show()
	# test_X, test_y = preprocess(test_df)
	print("Done with test")