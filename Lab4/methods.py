import numpy as np

# Функция сигмоиды
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Предотвращение переполнения
    return 1 / (1 + np.exp(-z))

# Функция вычисления стоимости
def compute_cost(X, y, w):
    m = len(y)
    z = np.dot(X, w)
    predictions = sigmoid(z)
    epsilon = 1e-15  # Для избежания логарифма от 0
    cost = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return cost

# Функция вычисления градиента
def compute_gradient(X, y, w):
    m = len(y)
    z = np.dot(X, w)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y)) / m
    return gradient

# Градиентный спуск
def gradient_descent(X, y, alpha, epochs):
    w = np.zeros(X.shape[1])  # Инициализация весов нулями
    for epoch in range(epochs):
        gradient = compute_gradient(X, y, w)
        w -= alpha * gradient
        if epoch % 1000 == 0 or epoch == epochs - 1:  # Лог каждые 1000 эпох
            cost = compute_cost(X, y, w)
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    return w

def manual_mean_and_std(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

def z_score_normalization(X):
    mean, std = manual_mean_and_std(X)

    # Проверка на стандартное отклонение, равное нулю
    std[std == 0] = 1  # Если std = 0, заменяем на 1, чтобы избежать деления на ноль

    X_norm = (X - mean) / std
    return X_norm, mean, std