# импорт необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from joblib import load

knn = load('knn_1') 

x_test = pd.read_csv('x_test.csv', index_col='index')
y_test = pd.read_csv('y_test.csv', index_col='index')

model = knn.predict(x_test)

print('Model test f1-score is: ',f1_score(y_test, model))
