from common import *
import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def load_cifar10_dataset():
  (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

  # Normalize pixel values to be between 0 and 1
  X_train, X_test = X_train / 255.0, X_test / 255.0

  return X_train, y_train, X_test, y_test