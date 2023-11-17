import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
'''Takes in confusion matrix output from model scripts and generates display of confusion matrix
Model scripts must be updated to include code to output confusion matrices'''

def cm_build(cm_path):
    cm = [[0]*15]*15  # create 15x15 array filled with 0s

    for user in range(1,16):
        # Open the file
        with open(f'{str(cm_path)}/user{user}_confusion_matrix.txt', 'r') as file:
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
    cm = np.array(cm_build(path))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    disp.plot(cmap='Blues')
    plt.show()


if __name__ == '__main__':
    model_output_path = 'model-outputs/knn/kvalue2'
    # model_output_path = 'model-outputs/random-forest/max-depth-7'
    # model_output_path = 'model-outputs/svc'
    cm_display(model_output_path)
