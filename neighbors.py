from common import *
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor

def train_kneigh_classif(k_values, X_train, y_train, X_val, y_val, X_test, y_test, method='uniform'):
  error_train = []
  error_val = []
  error_test = []
  
  best_val_error = math.inf

  for k in k_values:  
    knn = KNeighborsClassifier(n_neighbors=k, weights=method)

    knn.fit(X_train, y_train)

    predict_train = knn.predict(X_train)
    predict_val = knn.predict(X_val)
    predict_test = knn.predict(X_test)

    actual_train_error = 1 - accuracy_score(y_train, predict_train)
    actual_val_error = 1 - accuracy_score(y_val, predict_val)
    actual_test_error = 1 - accuracy_score(y_test, predict_test)

    error_train.append(actual_train_error)
    error_val.append(actual_val_error)
    error_test.append(actual_test_error)

    if actual_val_error < best_val_error:
      best_val_error = actual_val_error
      best_k = k
      best_knn = deepcopy(knn)

  return best_knn, best_k, error_train, error_val, error_test

def train_kneigh_regress(k_values, X_train, y_train, X_val, y_val, X_test, y_test, method='uniform'):
  error_train = []
  error_val = []
  error_test = []

  best_val_error = math.inf

  for k in k_values:  
    knn = KNeighborsRegressor(n_neighbors=k, weights=method)

    knn.fit(X_train, y_train)

    predict_train = knn.predict(X_train)
    predict_val = knn.predict(X_val)
    predict_test = knn.predict(X_test)

    actual_train_error = sk.metrics.mean_squared_error(y_train, predict_train)
    actual_val_error = sk.metrics.mean_squared_error(y_val, predict_val)
    actual_test_error = sk.metrics.mean_squared_error(y_test, predict_test)

    error_train.append(actual_train_error)
    error_val.append(actual_val_error)
    error_test.append(actual_test_error)

    if actual_val_error < best_val_error:
      best_val_error = actual_val_error
      best_k = k
      best_knn = deepcopy(knn)

  return best_knn, best_k, error_train, error_val, error_test

def train_radiusneigh_classif(r_values, X_train, y_train, X_val, y_val, X_test, y_test, method='uniform'):
  error_train = []
  error_val = []
  error_test = []
  
  best_val_error = math.inf

  for r in r_values:  
    knn = RadiusNeighborsClassifier(radius = r, weights=method)

    knn.fit(X_train, y_train)

    predict_train = knn.predict(X_train)
    predict_val = knn.predict(X_val)
    predict_test = knn.predict(X_test)

    actual_train_error = 1 - accuracy_score(y_train, predict_train)
    actual_val_error = 1 - accuracy_score(y_val, predict_val)
    actual_test_error = 1 - accuracy_score(y_test, predict_test)

    error_train.append(actual_train_error)
    error_val.append(actual_val_error)
    error_test.append(actual_test_error)

    if actual_val_error < best_val_error:
      best_val_error = actual_val_error
      best_r = r
      best_knn = deepcopy(knn)

  return best_knn, best_r, error_train, error_val, error_test 

def train_radiusneigh_regress(r_values, X_train, y_train, X_val, y_val, X_test, y_test, method='uniform'):
  error_train = []
  error_val = []
  error_test = []

  best_val_error = math.inf

  for r in r_values:  
    knn = RadiusNeighborsRegressor(radius = r, weights=method)

    knn.fit(X_train, y_train)

    predict_train = knn.predict(X_train)
    predict_val = knn.predict(X_val)
    predict_test = knn.predict(X_test)

    actual_train_error = sk.metrics.mean_squared_error(y_train, predict_train)
    actual_val_error = sk.metrics.mean_squared_error(y_val, predict_val)
    actual_test_error = sk.metrics.mean_squared_error(y_test, predict_test)

    error_train.append(actual_train_error)
    error_val.append(actual_val_error)
    error_test.append(actual_test_error)

    if actual_val_error < best_val_error:
      best_val_error = actual_val_error
      best_r = r
      best_knn = deepcopy(knn)

  return best_knn, best_r, error_train, error_val, error_test