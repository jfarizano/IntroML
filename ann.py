import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

def create_net_classifier(N2, eta, alfa, epochs):
  classif = MLPClassifier(hidden_layer_sizes=(N2,), activation='logistic', 
                       solver='sgd', alpha=0.0, batch_size=1, learning_rate='constant', 
                       learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,
                       tol=0.0,warm_start=True,max_iter=epochs)
  return classif

def create_net_regressor(N2, eta, alfa, epochs):
  regr = MLPRegressor(hidden_layer_sizes=(N2,), activation='logistic', 
                    solver='sgd', alpha=0.0, batch_size=1, learning_rate='constant', 
                    learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,
                    tol=0.0,warm_start=True,max_iter=epochs)
  return regr

#función que entrena una red ya definida previamente "evaluaciones" veces, cada vez entrenando un número de épocas elegido al crear la red y midiendo el error en train, validación y test al terminar ese paso de entrenamiento. 
#Guarda y devuelve la red en el paso de evaluación que da el mínimo error de validación
#entradas: la red, las veces que evalua, los datos de entrenamiento y sus respuestas, de validacion y sus respuestas, de test y sus respuestas
#salidas: la red entrenada en el mínimo de validación, los errores de train, validación y test medidos en cada evaluación
def train_net(net, evals, X_train, y_train, X_val, y_val, X_test, y_test):
     #mi código
     net.fit(X_train, y_train)
     #más código
     return best_net, error_train, error_val, error_test