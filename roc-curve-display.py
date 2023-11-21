import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# def roc_import(roc_path):  # TODO: this is just the script from the confusion matrix display script, i was basing the form off of this but do as you see fit
#     # SHOULDN'T NEED THIS
#     #cm = [[0]*15]*15  # create 15x15 array filled with 0s
#
#     for user in range(1,16):
#         # Open the file
#         with open(f'{str(roc_path)}/user{user}_pred.txt', 'r') as file:
#             # Read the contents of the file
#             content = file.read()
#         # Remove unnecessary characters and split the content into rows
#         content = content.replace('[','')
#         rows = content.split(']')
#
#         user_cm = [[int(num) for num in row.split()] for row in rows]
#         user_cm = user_cm[:15]
#         cm[user - 1] = user_cm[user - 1]
#
#     return cm


# def cm_display(path):
#     # define text file to open
#     roc = np.array(roc_import(path))
#
#     fpr, tpr, thresholds = metrics.roc_curve()
#     disp = RocCurveDisplay()
#     disp.plot(cmap='Blues')
#     plt.show()



def roc_display(path, model_name):

    # Read all testing data into 2 arrays X and y
    X = np.array
    y = np.array

    X = np.asarray(pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(1) + '.csv'))
    y = np.asarray(pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(1) + '.csv'))
    for user in range(2, 16):
        X = np.concatenate([X, np.asarray(
            pd.read_csv('processed-feature-data/testing-data/X_testing_data_user' + str(user) + '.csv'))])
        y = np.concatenate([y, np.asarray(
            pd.read_csv('processed-feature-data/testing-data/y_testing_data_user' + str(user) + '.csv'))])

    # Read all predictions into 1 array
    with open(str(path) + '/user' + str(1) + '_pred.txt', 'r') as file:
        content = file.read()
        content = content.replace('[', '')
        content = content.replace(']', '')
        rows = content.split(', ')
        pred = np.asarray([int(float(num)) for num in rows])

    for user in range(2, 16):
        with open(str(path) + '/user' + str(user) + '_pred.txt', 'r') as file:
            content = file.read()
        content = content.replace('[', '')
        content = content.replace(']', '')
        rows = content.split(', ')
        user_pred = [int(float(num)) for num in rows]
        pred = np.concatenate([pred, np.asarray(user_pred)])

    # Split data and calculate fpr, tpr, etc...
    X_test = X
    y_test = y

    y_test_bin = label_binarize(y_test, classes=list(range(1, 16)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    new_pred = np.array([[0] * 15] * 228199)

    for i in range(len(new_pred)):
        new_pred[i][pred[i] - 1] = 1

    for i in range(15):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], new_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot results
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab20', 15)

    for i in range(15):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2, label=f'ROC curve (class {i + 1}) (area = {roc_auc[i]:.2f})')

    # plt.plot(fpr[0], tpr[0], color=colors(0), lw=2, label=f'ROC curve (class {0+1}) (area = {roc_auc[0]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # plot dotted line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for ' + str(model_name))
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    # kNN
    # model_output_path = 'model-outputs/knn/kvalue2'
    # model_name = 'k-NN'

    # RF
    model_output_path = 'model-outputs/random-forest/max-depth-6'
    model_name = 'Random Forest'

    # SVC
    # model_output_path = 'model-outputs/svc'
    # model_name = 'SVC'

    roc_display(model_output_path, model_name)
