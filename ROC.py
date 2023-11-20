# Code for displaying the ROC curve

'''
Need a way to binarize the data and calculate the ROC curve for each user.
Sklearn's built-in roc_curve method only works with data so we'll have to
adapt to this.
'''

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

roc = roc_curve(y_test, pred, labels)

# Pseudocode
    # for i in range(1,16):
        # i = 1 --> convert I values to binary true value
        # sum array - i = 0 --> everything else is binary false value

