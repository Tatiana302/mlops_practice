# импорт необходимых библиотек
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = df.values # масштабирование данных

x, y = df[:, :-1], df[:, -1]
x.shape, y.shape # параметры ввода и вывода

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23) # разделим набор данных на обучающий и тестовый выборки

# предобработка данных
standart_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    
    ('scaler', StandardScaler())
]) 

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
x_test = pd.DataFrame(x_test)

x_test.to_csv('x_test.csv', index_label='index')
y_test.to_csv('y_test.csv', index_label='index') 
x_train.to_csv('x_train.csv', index_label='index')
y_train.to_csv('y_train.csv', index_label='index') 

print('data_preprocessing исполнен')
