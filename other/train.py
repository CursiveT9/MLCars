import pandas as pd
import random

df = pd.read_csv('../data/all_regions_trimmed_1000.csv')

useful_columns = ['year', 'mileage', 'power', 'engineDisplacement', 'fuelType', 'bodyType', 'brand', 'price']
df = df[useful_columns]

df = pd.get_dummies(df, columns=['fuelType', 'bodyType', 'brand'], drop_first=True)

X = df.drop('price', axis=1)
Y = df['price']

print("Обработанный датасет (первые 10 строк):")
print(df.head(10))

pd.set_option('display.max_columns', None)

print("\nВходные значения (X):")
print(X.head(1))

print("\nОжидаемое значение (Y):")
print(Y.head(1))

pd.set_option('display.max_columns', 7)

data_array = df.to_numpy()
n = data_array.shape[0]
for i in range(n - 1, 0, -1):
    j = random.randint(0, i)
    data_array[i], data_array[j] = data_array[j], data_array[i]
shuffled_df = pd.DataFrame(data_array, columns=df.columns)
print("\nПеремешанный датасет (первые 10 строк):")
print(shuffled_df.head(10))