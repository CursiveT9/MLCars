import numpy as np

def computeCost(x, y, theta):
    m = len(y)  # количество обучающих примеров
    predictions = theta[0] + theta[1] * x  # предсказания модели
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))  # вычисление ошибки
    return cost

# def computeCost(X, y, theta):
#     m = len(y)  # количество обучающих примеров
#     predictions = X.dot(theta)  # предсказания модели
#     cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))  # вычисление ошибки
#     return cost