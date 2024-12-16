import numpy as np
from Lab1.computeCost import computeCost

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)  # Количество обучающих примеров
    J_history = np.zeros(num_iters)  # Массив для хранения значений функции стоимости

    for i in range(num_iters):

        delta_0 = 0
        delta_1 = 0

        for j in range(m):
            prediction = theta[0] + theta[1] * x[j]  # Предсказание для j-го примера
            error = prediction - y[j]  # Ошибка для j-го примера
            delta_0 += error  # Накопление ошибки для theta_0
            delta_1 += error * x[j]  # Накопление ошибки для theta_1

        # Обновление параметров theta
        theta[0] = theta[0] - (alpha / m) * delta_0
        theta[1] = theta[1] - (alpha / m) * delta_1

        # Запись текущего значения функции стоимости
        J_history[i] = computeCost(x, y, theta)

    return theta, J_history
