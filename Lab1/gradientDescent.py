import numpy as np
from computeCost import computeCost

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)  # Массив для хранения значений функции стоимости

    for i in range(num_iters):
        predictions = theta[0] + theta[1] * x  # Предсказания модели
        theta[0] = theta[0] - (alpha / m) * np.sum(predictions - y)  # Обновление theta_0
        theta[1] = theta[1] - (alpha / m) * np.sum((predictions - y) * x)  # Обновление theta_1
        J_history[i] = computeCost(x, y, theta)  # Запись значения ошибки на текущей итерации

    return theta, J_history

# def gradientDescent(X, y, theta, alpha, num_iters):
#     m = len(y)
#     J_history = np.zeros(num_iters)  # Массив для хранения значений функции стоимости
#
#     for i in range(num_iters):
#         prediction = X.dot(theta)  # Предсказания модели
#         theta = theta - (alpha / m) * X.T.dot(prediction - y)  # Обновление параметров theta
#         J_history[i] = computeCost(X, y, theta)  # Запись значения ошибки на текущей итерации
#
#     return theta, J_history