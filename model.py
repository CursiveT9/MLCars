import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Шаг 1: Загрузка данных
data_path = 'data/all_regions_trimmed1000.csv'
df = pd.read_csv(data_path)

# Шаг 2: Предварительная обработка
df = df.dropna(subset=['price'])  # Убираем строки без цены
df = df.drop(columns=['name', 'location', 'link', 'description', 'parse_date'])  # Убираем ненужные столбцы

# Преобразуем категориальные переменные в числовые
df = pd.get_dummies(df, columns=['brand', 'bodyType', 'color', 'fuelType', 'transmission', 'vehicleConfiguration', 'engineName'], drop_first=True)

# Шаг 3: Выбор признаков и целевой переменной
X = df[['year', 'mileage', 'power', 'engineDisplacement'] + [col for col in df.columns if 'brand_' in col or 'bodyType_' in col]]
y = df['price']

# Шаг 4: Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 5: Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Шаг 6: Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
