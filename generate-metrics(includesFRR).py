import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


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


def generate_metrics(path):
    cm = cm_build(path)

    TP = [0] * 15
    FP = [0] * 15
    FN = [0] * 15
    precision = [0] * 15  # (TP/(TP+FP))
    recall = [0] * 15  # (TP/(TP+FN))
    f1 = [0] * 15  # (2 * ((P*R)/(P+R)))
    accuracy = [0] * 15  # (TP + TN) / (TP + TN + FP + FN)
    frr = [0] * 15


    for user in range(15):
        TP[user] = cm[user][user]
        FN[user] = sum(cm[user]) - TP[user]
        for row in range(15):
            if row == user:
                continue
            else:
                FP[user] += cm[row][user]
        precision[user] = TP[user] / (TP[user] + FP[user])
        recall[user] = TP[user] / (TP[user] + FN[user])
        f1[user] = 2 * ((precision[user] * recall[user]) / (precision[user] + recall[user]))
        accuracy[user] = TP[user] / (TP[user] + FP[user] + FN[user])
        # far = FP / (FP + TN)
        frr[user] = FN[user] / (FN[user] + TP[user])

    return [precision, recall, f1, accuracy, frr]


def output_metrics(metrics, path):
    with open(path + '/metrics.txt', mode="w") as f:
        f.write('user   precision   recall   f1-score   accuracy   frr\n')
        for user in range(15):
            f.write('{:>3}'.format(str(user + 1)))
            f.write('{:10.2f}'.format(metrics[0][user]))
            f.write('{:11.2f}'.format(metrics[1][user]))
            f.write('{:10.2f}'.format(metrics[2][user]))
            f.write('{:10.2f}'.format(metrics[3][user]))
            f.write('{:10.2f}\n'.format(metrics[4][user]))
        f.write('\n')
        f.write('{:>3}'.format('avg'))
        f.write('{:10.2f}'.format(sum(metrics[0]) / 15))
        f.write('{:11.2f}'.format(sum(metrics[1]) / 15))
        f.write('{:10.2f}'.format(sum(metrics[2]) / 15))
        f.write('{:10.2f}'.format(sum(metrics[3]) / 15))
        f.write('{:10.2f}\n'.format(sum(metrics[4]) / 15))


if __name__ == '__main__':
    # model_output_path = 'model-outputs/knn/kvalue2'
    # model_output_path = 'model-outputs/random-forest/max-depth-6'
    model_output_path = 'model-outputs/svc'
    metrics = generate_metrics(model_output_path)
    output_metrics(metrics, model_output_path)