# File for generating ROC Curves

# Going to first just try for 1 user hard coded, then will expand

# What we need:
#   - true positives
#   - false negatives
#   - true positive rate
#   - true negative rate

# Once we have these four metrics, we can calculate the ROC curve

# imports
from confusion_matrix_display import cm_build
from sklearn import metrics
import numpy as np

# Start with user 1:

cm = cm_build('model-outputs/knn/kvalue2')
print(str(cm) + '\n' + '--------------------------------------------------------------')

# Print user 1's true postives
tp = cm[0][0]
print(tp)
# Should be 16092

# todo: change to be more modular
column_total = sum(row[0] for row in cm)
print(column_total)
# should be 21195
fp = column_total - tp
print(fp)
# Should be 5103

tpr = tp / (tp+fp)
fpr = fp / (tp+fp)

print(tpr)
print(fpr)

roc_auc = metrics.auc(fpr,tpr)