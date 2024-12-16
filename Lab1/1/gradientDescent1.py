import numpy as np
from computeCost1 import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)  # Массив для хранения значений функции стоимости

    for i in range(num_iters):
        prediction = X.dot(theta)  # Предсказания модели
        theta = theta - (alpha / m) * X.T.dot(prediction - y)  # Обновление параметров theta
        J_history[i] = computeCost(X, y, theta)  # Запись значения ошибки на текущей итерации

    return theta, J_history