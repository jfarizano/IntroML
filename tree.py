import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# A partir de un dataframe, lo tomo como conjunto de entrenamiento y entreno
# un árbol a partir de él
def train_tree(df):
  X_train, y_train = df[[0, 1]], df['Class']
  clf = DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=0.005,random_state=0,min_samples_leaf=5)
  clf.fit(X_train, y_train)

  return clf

# A partir de un conjunto de datos y un árbol de decisión, predigo sobre 
# el conjunto dado
def predict_tree(df_test, clf):
  X_test = df_test[[0, 1]]
  predict = clf.predict(X_test)
  df_predict = df_test.copy(deep = True)
  df_predict['Class'] = predict
  return df_predict

# Defino una función para graficar la cantidad de nodos de los árboles resultantes
# a partir de los datos de entrenamiento
def graph_nodes_count(count_df):
  fig, ax = plt.subplots(figsize=(10, 10))

  colors = ["blue", "orange"]
  classes = pd.unique(count_df['Class'])

  for (p, c) in zip(classes, colors):
   df = count_df[count_df['Class'] == p]
   df = df.groupby('n').mean()
   df = df.reset_index()
   plt.plot(df['n'], df['Nodes'], color=c, marker='o')

  ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

  plt.legend(classes, ncol = 4)
  plt.xlabel('n', size=14, labelpad=15)
  plt.ylabel('Nodes', size=14, labelpad=15)
