from functions import warmUpExercise, plotData, computeCost, gradientDescent
from tools import loadtxt, zeros, ones, column_stack, dot
import matplotlib.pyplot as plt

# 1. Загрузка данных из файла ex1data1.txt
data = loadtxt('ex1data1.txt', delimiter=',')
X_values = [row[0] for row in data]  # Количество автомобилей в городе
y = [row[1] for row in data]  # Прибыль СТО
m = len(y)  # Количество примеров

# 2. Визуализация данных
print("Отображаем исходные данные:")
plotData(X_values, y)

# 3. Добавляем колонку единиц к X (чтобы учесть theta_0)
X = [[1, X_values[i]] for i in range(m)]  # Матрица признаков размером m x 2

# 4. Инициализация параметров
theta = zeros(2)  # Начальные значения параметров
iterations = 1500  # Количество итераций градиентного спуска
alpha = 0.01  # Скорость обучения

# 5. Вычисление начальной функции стоимости
initial_cost = computeCost(X, y, theta)
print(f"Начальная функция стоимости: {initial_cost}")

# 6. Запуск градиентного спуска
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

print(f"Найденные параметры theta после градиентного спуска: {theta}")
print(f"Функция стоимости после градиентного спуска: {J_history[-1]}")

# 7. Построение графика линейной регрессии
plt.figure()
plt.scatter(X_values, y, color='red', marker='x', label='Обучающие данные')

# Предсказанные значения
h_values = [dot(X[i], theta) for i in range(m)]
plt.plot(X_values, h_values, label='Линейная регрессия')

plt.xlabel('Количество автомобилей в городе')
plt.ylabel('Прибыль СТО')
plt.legend()
plt.title('Линейная регрессия для данных СТО')
plt.grid(True)
plt.show()

# 8. Прогнозирование прибыли для населённых пунктов с 35,000 и 70,000 автомобилей
predict1 = dot([1, 3.5], theta) * 10000  # Прогноз для 35,000 автомобилей
predict2 = dot([1, 7], theta) * 10000    # Прогноз для 70,000 автомобилей

print(f"Прогнозируемая прибыль для города с 35,000 автомобилей: ${predict1:.2f}")
print(f"Прогнозируемая прибыль для города с 70,000 автомобилей: ${predict2:.2f}")
