import numpy as np
import matplotlib.pyplot as plt


def feature_normalize(X):
    """Нормализует признаки, возвращает нормализованные данные, среднее и стандартное отклонение."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost(X, y, theta):
    """Вычисляет функцию стоимости для линейной регрессии."""
    m = len(y)
    predictions = X @ theta
    sq_errors = (predictions - y) ** 2
    J = (1 / (2 * m)) * np.sum(sq_errors)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """Выполняет градиентный спуск для оптимизации параметров."""
    m = len(y)
    J_history = []

    for _ in range(num_iters):
        error = (X @ theta) - y
        theta -= (alpha / m) * (X.T @ error)
        J_history.append(compute_cost(X, y, theta))

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