import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/Tatiana302/dataset/main/diabetes.csv')
df.to_csv('df.csv')
print('data_creation исполнен')
