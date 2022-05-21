import math
from copy import deepcopy
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

def train_Categorical_NB(n_bins, X_train, y_train, X_val, y_val, X_test, y_test):
  best_val_error = math.inf
  
  errors = []

  for bins in n_bins:
    kbdisc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    kbdisc.fit(X_train)

    X_train_disc = kbdisc.transform(X_train.copy())
    X_val_disc = kbdisc.transform(X_val.copy())
    X_test_disc = kbdisc.transform(X_test.copy())

    clf = CategoricalNB(min_categories=bins)
    clf.fit(X_train_disc, y_train)

    predict_train = clf.predict(X_train_disc)
    predict_val = clf.predict(X_val_disc)
    predict_test= clf.predict(X_test_disc)

    actual_train_error = 1 - accuracy_score(y_train, predict_train)
    actual_val_error = 1 - accuracy_score(y_val, predict_val)
    actual_test_error = 1 - accuracy_score(y_test, predict_test)

    errors.append([actual_train_error, bins, "Train error"])
    errors.append([actual_val_error, bins, "Validation error"])
    errors.append([actual_test_error, bins, "Test error"])

    if actual_val_error < best_val_error:
      best_val_error = actual_val_error
      best_bins = bins
      best_clf = deepcopy(clf)
      best_kbdisc = deepcopy(kbdisc)

  errors_df = pd.DataFrame(errors, columns = ["Error", "Bins", "Class"])

  return best_bins, best_clf, best_kbdisc, errors_df
