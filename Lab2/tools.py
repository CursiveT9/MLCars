import numpy as np
import matplotlib.pyplot as plt

def feature_normalize(X):
    """Нормализует признаки, возвращает нормализованные данные, среднее и стандартное отклонение."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# def feature_normalize(X):
#     mean, std = manual_mean_and_std(X)
#     X_norm = (X - mean) / std
#     return X_norm, mean, std
#
# def manual_mean_and_std(X):
#     n = len(X)  # Количество элементов
#     mean = np.sum(X) / n  # Среднее
#     std_dev = np.sqrt(np.sum((X - mean) ** 2) / n) # Стандартная
#     return mean, std_dev

def computeCost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors, errors)
    return cost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        # Векторизованное обновление параметров
        errors = X @ theta - y
        theta -= (alpha / m) * (X.T @ errors)
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history

def normal_equation(X, y):
    """Выполняет аналитическое решение для нахождения параметров."""
    return np.linalg.inv(X.T @ X) @ X.T @ y


def plot_cost_function(J_history):
    """Визуализирует сходимость функции стоимости."""
    plt.plot(range(len(J_history)), J_history, 'b-')
    plt.xlabel('Итерации')
    plt.ylabel('Функция стоимости J')
    plt.title('Сходимость функции стоимости')
    plt.show()


def plot_predictions_comparison(y, X, pred_gd, pred_analytic):
    """Генерирует два 3D-графика сравнения предсказаний (градиентный спуск и аналитическое решение)."""

    # Создаем фигуру для двух подграфиков
    fig = plt.figure(figsize=(12, 8))

    # График для градиентного спуска
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], y, color='blue', label='Реальные данные', alpha=0.6)
    ax1.plot_trisurf(X[:, 0], X[:, 1], pred_gd, color='red', alpha=0.5, label='Градиентный спуск')
    ax1.set_xlabel('Обороты двигателя')
    ax1.set_ylabel('Количество передач')
    ax1.set_zlabel('Цена')
    ax1.set_title('Предсказания с помощью градиентного спуска')
    ax1.legend()

    # График для аналитического решения
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X[:, 0], X[:, 1], y, color='blue', label='Реальные данные', alpha=0.6)
    ax2.plot_trisurf(X[:, 0], X[:, 1], pred_analytic, color='green', alpha=0.5, label='Аналитическое решение')
    ax2.set_xlabel('Обороты двигателя')
    ax2.set_ylabel('Количество передач')
    ax2.set_zlabel('Цена')
    ax2.set_title('Предсказания с помощью аналитического решения')
    ax2.legend()

    # Покажем оба графика
    plt.show()

def plot_predictions_comparison2(y, X, pred_gd, pred_analytic):
    """Генерирует 3D-график сравнения реальных данных и предсказанных значений."""

    # Реальные значения (X[:, 0] - обороты двигателя, X[:, 1] - количество передач)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Реальные данные
    ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Реальные данные', alpha=0.6)

    # Плоскость для предсказаний (градиентный спуск)
    ax.plot_trisurf(X[:, 0], X[:, 1], pred_gd, color='red', alpha=0.5, label='Градиентный спуск')

    # Плоскость для предсказаний (аналитическое решение)
    ax.plot_trisurf(X[:, 0], X[:, 1], pred_analytic, color='green', alpha=0.5, label='Аналитическое решение')

    # Настройки графика
    ax.set_xlabel('Обороты двигателя')
    ax.set_ylabel('Количество передач')
    ax.set_zlabel('Цена')

    ax.set_title('Сравнение предсказаний: Градиентный спуск vs Аналитическое решение')

    ax.legend()
    plt.show()


# График двух признаков (нормализованные и исходные данные)
def plot_features_comparison(X_original, X_normalized):
    plt.figure(figsize=(14, 6))

    # График исходных данных
    plt.subplot(1, 2, 1)
    plt.scatter(X_original[:, 0], X_original[:, 1], c='blue', alpha=0.7, edgecolor='k', label="Исходные данные")
    plt.title("Исходные данные (ненормализованные)", fontsize=14)
    plt.xlabel("Признак 1 (Обороты двигателя)", fontsize=12)
    plt.ylabel("Признак 2 (Количество передач)", fontsize=12)
    plt.grid(True)
    plt.legend()

    # График нормализованных данных
    plt.subplot(1, 2, 2)
    plt.scatter(X_normalized[:, 1], X_normalized[:, 2], c='red', alpha=0.7, edgecolor='k',
                label="Нормализованные данные")
    plt.title("Нормализованные данные", fontsize=14)
    plt.xlabel("Признак 1 (нормализованный)", fontsize=12)
    plt.ylabel("Признак 2 (нормализованный)", fontsize=12)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()