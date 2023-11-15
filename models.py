# Main file for processing the data for all of the models

# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import train_test_split

from typing import Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

# Need to be able to output everything to a txt file


def main():
    with open("final_output.txt", w):


if __name__ == "__main__":
    main()