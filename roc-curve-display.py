import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import numpy as np
from sklearn import metrics

'''Takes in roc curve output from model scripts and generates display of roc matrix
Model scripts must be updated to include code to output roc curve'''

def roc_import(roc_path):  # TODO: this is just the script from the confusion matrix display script, i was basing the form off of this but do as you see fit
    # SHOULDN'T NEED THIS
    #cm = [[0]*15]*15  # create 15x15 array filled with 0s

    for user in range(1,16):
        # Open the file
        with open(f'{str(roc_path)}/user{user}_confusion_matrix.txt', 'r') as file:
            # Read the contents of the file
            content = file.read()
        # Remove unnecessary characters and split the content into rows
        content = content.replace('[','')
        rows = content.split(']')

        user_cm = [[int(num) for num in row.split()] for row in rows]
        user_cm = user_cm[:15]
        cm[user - 1] = user_cm[user - 1]

    return cm


def cm_display(path):
    # define text file to open
    roc = np.array(roc_import(path))

    fpr, tpr, thresholds = metrics.roc_curve()
    disp = RocCurveDisplay()
    disp.plot(cmap='Blues')
    plt.show()


if __name__ == '__main__':
    model_output_path = 'model-outputs/knn/kvalue2'
    cm_display(model_output_path)
