import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Lab2.normalize import z_score_normalization

# Загрузка данных из CSV-файла
data = pd.read_csv('../../data/all_regions_trimmed_400000.csv')

# Выбираем все столбцы с типом данных object (категориальные данные)
categorical_columns = data.select_dtypes(include=[object]).columns
# Создаем словарь для хранения соответствий между категориями и их числовыми кодами
category_mappings = {}
# Проходим по каждому категориальному столбцу
for col in categorical_columns:
    # Преобразуем категории в числовые коды и сохраняем соответствия
    data[col], mapping = pd.factorize(data[col])
    category_mappings[col] = mapping

# Преобразование года, пробега и мощности в числовые значения
data['year'] = data['year'].astype(float)
data['mileage'] = data['mileage'].astype(float)
data['power'] = data['power'].astype(float)
data['engineDisplacement'] = data['engineDisplacement'].astype(float)

# Создание бинарной целевой переменной
# Если цена больше 1 000 000, то значение 1, иначе 0
data['is_expensive'] = (data['price'] > 1000000).astype(int)

# Отделение признаков от целевой переменной
# Убираем столбцы 'price' и 'is_expensive' из признаков (X)
X = data.drop(columns=['price', 'is_expensive'])
# Целевая переменная (y) — это столбец 'is_expensive'
y = data['is_expensive']

# Нормализация по среднему и стандартному отклонению (z-score normalization)
X, mean_values, std_values = z_score_normalization(X)

# Разделение данных на обучающую и тестовую выборки
train_size = int(0.8 * len(data))
# Разделяем данные на обучающую и тестовую выборки
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Добавление столбца
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Нормализация по среднему и стандартному отклонению (z-score normalization)
def z_score_normalization(X):
    mean, std = manual_mean_and_std(X)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def manual_mean_and_std(X):
    n = len(X)  # Количество элементов
    mean = np.sum(X) / n  # Среднее
    std_dev = np.sqrt(np.sum((X - mean) ** 2) / n) # Стандартное отклонение
    return mean, std_dev

# Сигмоидная функция (функция активации)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Предсказание вероятности
def predict(X, weights):
    return sigmoid(np.dot(X, weights))

# Функция для вычисления стоимости (ошибки) модели
def cost_function(X, y, weights):
    m = len(y)  # Количество примеров в выборке
    predictions = predict(X, weights)  # Предсказанные вероятности
    # Вычисляем логистическую потерю
    cost = (-1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Функция для оптимизации весов модели с использованием градиентного спуска
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)  # Количество примеров в выборке
    cost_history = np.zeros(iterations)  # Массив для хранения истории стоимости

    for i in range(iterations):
        predictions = predict(X, weights)  # Предсказанные вероятности
        # Обновляем веса с использованием градиента
        weights -= (learning_rate/m) * np.dot(X.T, predictions - y)
        # Сохраняем текущее значение функции стоимости
        cost_history[i] = cost_function(X, y, weights)

    return weights, cost_history

# Инициализация весов
weights = np.zeros(X_train.shape[1])

# Выполняем градиентный спуск для обучения модели
weights, cost_history = gradient_descent(X_train, y_train, weights, learning_rate=0.1, iterations=700)

# Создаем словарь для хранения модели и других необходимых данных
model_data = {
    'model': weights,  # Обученные веса
    'category_mappings': category_mappings,  # Соответствия категорий и числовых кодов
    'mean_values': mean_values,  # Средние значения для нормализации
    'std_values': std_values  # Стандартные отклонения для нормализации
}

# Сохраняем модель и данные в файл с использованием joblib
joblib.dump(model_data, 'logistic_model.pkl')

# Предсказываем вероятности для тестовой выборки
y_pred = predict(X_test, weights)
# Преобразуем вероятности в классы (0 или 1) с порогом 0.5
y_pred_class = (y_pred >= 0.5).astype(int)

# Вычисляем точность модели как долю правильных предсказаний
accuracy = np.mean(y_pred_class == y_test)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Строим график изменения функции стоимости в процессе обучения
plt.plot(range(700), cost_history)
plt.xlabel("Итерации")
plt.ylabel("Функция стоимости")
plt.title("Сходимость функции стоимости")
plt.show()

# Строим ящик с усами для визуализации распределения признаков
X.plot(kind='box', figsize=(12, 8))
plt.title("Ящик с усами признаков")
plt.xlabel("Признаки")
plt.ylabel("Нормализованные значения")
plt.show()