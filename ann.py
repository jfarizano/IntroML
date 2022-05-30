from common import *
from sklearn.base import is_classifier
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def create_net_classifier(N2, eta, alfa, epochs, gamma = 0.0, batch = 1):
  classif = MLPClassifier(hidden_layer_sizes=(N2,), activation='logistic', 
                       solver='sgd', alpha=gamma, batch_size=batch, learning_rate='constant', 
                       learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,
                       tol=0.0,warm_start=True,max_iter=epochs)
  return classif

def create_net_regressor(N2, eta, alfa, epochs, gamma = 0.0, batch = 1):
  regr = MLPRegressor(hidden_layer_sizes=(N2,), activation='logistic', 
                    solver='sgd', alpha=gamma, batch_size=batch, learning_rate='constant', 
                    learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,
                    tol=0.0,warm_start=True,max_iter=epochs)
  return regr


#función que entrena una red ya definida previamente "evaluaciones" veces, cada vez entrenando un número de épocas elegido al crear la red y midiendo el error en train, validación y test al terminar ese paso de entrenamiento. 
#Guarda y devuelve la red en el paso de evaluación que da el mínimo error de validación
#entradas: la red, las veces que evalua, los datos de entrenamiento y sus respuestas, de validacion y sus respuestas, de test y sus respuestas
#salidas: la red entrenada en el mínimo de validación, los errores de train, validación y test medidos en cada evaluación
def train_net(net, evals, X_train, y_train, X_val, y_val, X_test, y_test):
  error_train = []
  error_val = []
  error_test = []

  best_error_val = math.inf

  for i in range(evals):
    net.fit(X_train, y_train)

    predict_train = net.predict(X_train)
    predict_val = net.predict(X_val)
    predict_test = net.predict(X_test)

    if is_classifier(net):
      actual_error_train = sk.metrics.zero_one_loss(y_train, predict_train)
      actual_error_val = sk.metrics.zero_one_loss(y_val, predict_val)
      actual_error_test = sk.metrics.zero_one_loss(y_test, predict_test)
    else:
      actual_error_train = sk.metrics.mean_squared_error(y_train, predict_train)
      actual_error_val = sk.metrics.mean_squared_error(y_val, predict_val)
      actual_error_test = sk.metrics.mean_squared_error(y_test, predict_test)  

    error_train.append(actual_error_train)
    error_val.append(actual_error_val)
    error_test.append(actual_error_test)

    # Busco la mejor red
    if actual_error_val < best_error_val:
      best_error_val = actual_error_val
      best_net = deepcopy(net)

  return best_net, error_train, error_val, error_test