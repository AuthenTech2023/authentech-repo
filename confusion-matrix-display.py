import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


# ChatGPT read 2d list from file:
# # Open the file
# file_path = 'path/to/your/file.txt'
# with open(file_path, 'r') as file:
#     # Read the contents of the file
#     content = file.read()
#
# # Remove unnecessary characters and split the content into rows
# rows = content.strip('[]').split('\n')
#
# # Convert each row into a list of integers
# matrix = [[int(num) for num in row.split()] for row in rows]
#
# # Print the resulting 2-dimensional list
# for row in matrix:
#     print(row)





def cm_build(cm_path):
    # cm = np.array([[0]*15]*15)  # create 15x15 array filled with 0s
    cm = [[0]*15]*15

    for user in range(1,16):
        # user_cm = np.array(open(f'{str(cm_path)}/user{user}_confusion_matrix.txt', 'r').read())
        # user_cm = list(open(f'{str(cm_path)}/user{user}_confusion_matrix.txt', 'r').read())
        # Open the file
        with open(f'{str(cm_path)}/user{user}_confusion_matrix.txt', 'r') as file:
            # Read the contents of the file
            content = file.read()
        # Remove unnecessary characters and split the content into rows
        # content = content.strip('[')
        # content = content.strip(']')
        content = content.replace('[','')
        # content = content.replace(']','')
        # print(content)
        rows = content.split(']')
        # print(rows)
        # Convert each row into a list of integers
        user_cm = [[int(num) for num in row.split()] for row in rows]
        user_cm = user_cm[:15]
        print(f'user_cm: {user_cm}')
        # print(type(user_cm))
        # print(user_cm)
        cm[user - 1] = user_cm[user - 1]

    return cm


def cm_display(path):
    # define text file to open
    cm = np.array(cm_build(path))
    print(f'cm: {cm}')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    disp.plot(cmap='Blues')
    plt.show()

if __name__ == '__main__':
    model_output_path = 'model-outputs/knn/kvalue2'
    cm_display(model_output_path)