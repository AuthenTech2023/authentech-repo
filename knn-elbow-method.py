import numpy as np
from KNN import knn_model
from matplotlib import figure
import matplotlib as plt


print('Welcome to the Elbow Method Script!')
auth_user = input('Enter the Authorized User ID (1-15): ')
unauth_user = input('Enter the Unauthorized User ID (1-15): ')

error_rate = []

for i in range(1 ,40):
    pred_i, y_test = knn_model(auth_user, unauth_user, i)
    error_rate.append(np.mean(pred_i != y_test))

# Save output to file (just in case)
with open('elbow-output.txt', 'w') as f:
    for item in error_rate:
        f.write(str(item))

# create plot of results
# TODO: fix this, it should create a plot of error rates
#plt.figure(figsize=(10,6))

#plt.plot(range(1,40), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)

print(error_rate.index(min(error_rate)) + 1) # print k value with lowest error rate