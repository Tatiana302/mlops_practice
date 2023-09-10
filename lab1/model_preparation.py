# импорт необходимых библиотек
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from sklearn.pipeline import Pipeline

x_train = pd.read_csv('x_train.csv', index_col='index')
y_train = pd.read_csv('y_train.csv', index_col='index')

pipe = Pipeline([ 
    ('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=2)) # назначаем в качестве модели knn
])

pipe = pipe.fit(x_train, y_train)

dump(pipe, 'knn_1') # задаём название для модели

knn_mod = load('knn_1') # назначаем переменную и сохраняем

print('model_preparation исполнен')
