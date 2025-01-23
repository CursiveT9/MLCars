import joblib
import pandas as pd
import numpy as np

# Загрузка данных
data = pd.read_csv('../../data/all_regions_trimmed_400000.csv')

# Заполнение пропущенных значений в числовых столбцах средними значениями
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Преобразование категориальных столбцов в числовые
categorical_columns = data.select_dtypes(include=[object]).columns
category_mappings = {}
for col in categorical_columns:
    data[col], mapping = pd.factorize(data[col])
    category_mappings[col] = mapping

# Преобразование года, пробега и мощности в числовые значения
data['year'] = data['year'].astype(float)
data['mileage'] = data['mileage'].astype(float)
data['power'] = data['power'].astype(float)
data['engineDisplacement'] = data['engineDisplacement'].astype(float)

# Целевая переменная — цена
y = data['price']

# Отделение признаков от целевой переменной
X = data.drop(columns=['price'])

# Нормализация данных
mean_values = X.mean()  # Среднее значение по каждому столбцу
std_values = X.std()    # Стандартное отклонение по каждому столбцу

# Нормализуем данные
X = (X - mean_values) / std_values

# Разделение данных на обучающую и тестовую выборки
train_size = int(0.8 * len(data))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Линейная регрессия
def predict(X, weights):
    return np.dot(X, weights)

def cost_function(X, y, weights):
    m = len(y)
    predictions = predict(X, weights)
    cost = (1/(2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = predict(X, weights)
        weights -= (learning_rate/m) * np.dot(X.T, predictions - y)
        cost_history[i] = cost_function(X, y, weights)

    return weights, cost_history

# Добавление столбца единиц для смещения (базовый уровень)
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Инициализация весов
weights = np.zeros(X_train.shape[1])

# Обучение модели
weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate=0.01, iterations=1000)

# Сохранение модели и всех необходимых объектов
model_data = {
    'model': weights,
    'category_mappings': category_mappings,
    'mean_values': mean_values,
    'std_values': std_values
}

# Сохранение всех объектов в одном файле
joblib.dump(model_data, 'full_model.pkl')

# Предсказания на тестовой выборке
y_pred = predict(X_test, weights)

# Оценка модели (среднеквадратичная ошибка)
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse:.2f}")
me = np.mean((y_pred - y_test))
print(f"Mean Error: {me:.2f}")

# График сходимости функции стоимости
import matplotlib.pyplot as plt
plt.plot(range(1000), cost_history)
plt.xlabel("Итерации")
plt.ylabel("Функция стоимости")
plt.title("Сходимость функции стоимости")
plt.show()

# Box plot для визуализации распределения признаков
X.plot(kind='box', figsize=(12, 8))
plt.title("Ящик с усами признаков")
plt.xlabel("Признаки")
plt.ylabel("Нормализованные значения")
plt.show()
