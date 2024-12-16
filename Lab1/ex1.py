import numpy as np
import matplotlib.pyplot as plt

from Lab1.computeCost import computeCost
from plotData import plotData
from gradientDescent import gradientDescent

# Загрузка данных
data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:, 0]  # Количество автомобилей
y = data[:, 1]  # Прибыль СТО

# Добавление единичной переменной для theta_0 (для свободного члена)
m = len(x)

# Визуализация данных
plotData(x, y)

# Инициализация параметров
theta = np.zeros(2)
alpha = 0.01  # Шаг обучения
num_iters = 1500

# Вычисление начальной функции стоимости (с нулевыми параметрами)
initial_cost = computeCost(x, y, theta)
print(f"Начальная функция стоимости: {initial_cost}")

# Градиентный спуск
theta, J_history = gradientDescent(x, y, theta, alpha, num_iters)

# Печать оптимальных параметров
print(f"Оптимальные параметры: theta_0 = {theta[0]}, theta_1 = {theta[1]}")

# Вычисление функции стоимости после градиентного спуска
final_cost = computeCost(x, y, theta)
print(f"Функция стоимости после градиентного спуска: {final_cost}")

# Визуализация ошибки
plt.plot(range(num_iters), J_history, label='Функция стоимости')
plt.xlabel('Количество итераций')
plt.ylabel('Ошибка')
plt.title('Конвергенция функции стоимости')
plt.grid(True)
plt.show()

# Прогнозирование на основе полученных параметров
plt.scatter(x, y, c='red', label='Обучающие данные')
plt.plot(x, theta[0] + theta[1] * x, label='Апроксимирующая прямая', color='blue')  # Исправлено
plt.xlabel('Количество автомобилей')
plt.ylabel('Прибыль СТО')
plt.legend()
plt.title('Линейная регрессия: Прогноз прибыли')
plt.grid(True)
plt.show()


# Прогнозы для нескольких значений x
forecast_x_values = np.array([50, 100, 1500, 2000])  # Значения для которых делаем прогноз
forecast_y_values = theta[0] + theta[1] * forecast_x_values  # Прогнозы

# Вывод прогнозов
for x_value, y_value in zip(forecast_x_values, forecast_y_values):
    print(f"Прогноз для {x_value} автомобилей: {y_value:.2f} прибыли")