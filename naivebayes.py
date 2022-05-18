import math
from copy import deepcopy
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import CategoricalNB

# def train_Categorical_NB(n_bins, X_train, y_train, X_val, y_val, X_test, y_test):
#   best_val_error = math.inf
  
#   errors = []

#   for bins in n_bins:
#     disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
#     disc.fit(X_train)
#     X_train_disc = disc.transform(X_train.copy())

#     clf = CategoricalNB(min_categories=bins)
#     clf.fit(X_train_disc, y_train)
