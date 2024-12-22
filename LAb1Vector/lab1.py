import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]  # Количество автомобилей
y = data[:, 1]  # Прибыль СТО
m = len(y)  # Количество примеров

# Добавление единичной переменной для theta_0
X = np.stack((np.ones(m), X), axis=1)  # Добавляем столбец из единиц

# Визуализация данных
plt.scatter(X[:, 1], y, c='red', label='Обучающие данные')
plt.xlabel('Количество автомобилей')
plt.ylabel('Прибыль СТО')
plt.title('Данные')
plt.legend()
plt.grid(True)
plt.show()

# Функция вычисления стоимости (векторизованная версия)
def computeCost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    return cost

# Градиентный спуск (векторизованная версия)
def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        # Векторизованное обновление параметров
        errors = X @ theta - y
        theta -= (alpha / m) * (X.T @ errors)
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history

# Инициализация параметров
theta = np.zeros(2)
alpha = 0.02  # Шаг обучения, выбрал эмпирически(наблюдение изменений)
num_iters = 700
# alpha = 0.01  # Шаг обучения
# num_iters = 1500

# Вычисление начальной функции стоимости
initial_cost = computeCost(X, y, theta)
print(f"Начальная функция стоимости: {initial_cost}")

# Градиентный спуск
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Печать оптимальных параметров
print(f"Оптимальные параметры: theta_0 = {theta[0]}, theta_1 = {theta[1]}")

# Вычисление функции стоимости после градиентного спуска
final_cost = computeCost(X, y, theta)
print(f"Функция стоимости после градиентного спуска: {final_cost}")

# Визуализация ошибки
plt.plot(range(num_iters), J_history, label='Функция стоимости')
plt.xlabel('Количество итераций')
plt.ylabel('Ошибка')
plt.title('Конвергенция функции стоимости')
plt.grid(True)
plt.show()

# Визуализация данных и линии регрессии
plt.scatter(X[:, 1], y, c='red', label='Обучающие данные')
plt.plot(X[:, 1], X @ theta, label='Апроксимирующая прямая', color='blue')
plt.xlabel('Количество автомобилей')
plt.ylabel('Прибыль СТО')
plt.legend()
plt.title('Линейная регрессия: Прогноз прибыли')
plt.grid(True)
plt.show()

# Прогнозы для нескольких значений X
forecast_x_values = np.array([50, 100, 1500, 2000])
forecast_X = np.stack((np.ones(len(forecast_x_values)), forecast_x_values), axis=1)
forecast_y_values = forecast_X @ theta

# Вывод прогнозов
for x_value, y_value in zip(forecast_x_values, forecast_y_values):
    print(f"Прогноз для {x_value} автомобилей: {y_value:.2f} прибыли")
