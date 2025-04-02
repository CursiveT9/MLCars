from tools import *
import matplotlib.pyplot as plt

# 1. warmUpExercise
def warmUpExercise(n):
    """
    Возвращает единичную матрицу размером n x n.

    Аргументы:
    n -- размерность матрицы

    Возвращает:
    Единичная матрица n x n
    """
    # Способ 1: Используем кастомную функцию для единичной матрицы
    identity_matrix = eye(n)

    # Способ 2: Собственное вычисление единичной матрицы
    identity_matrix_custom = zeros((n, n))
    for i in range(n):
        identity_matrix_custom[i][i] = 1

    return identity_matrix, identity_matrix_custom


# 2. plotData
def plotData(X, y):
    """
    Строит график данных X и y с использованием scatter plot.

    Аргументы:
    X -- массив признаков (количество автомобилей в городах)
    y -- массив целевых значений (прибыль СТО)
    """
    plt.scatter(X, y, color='red', marker='x', label='Данные')
    plt.xlabel('Количество автомобилей в городе')
    plt.ylabel('Прибыль СТО (в 10,000$)')
    plt.title('Зависимость прибыли от числа автомобилей')
    plt.grid(True)
    plt.show()


# 3. computeCost
def computeCost(X, y, theta):
    """
    Вычисляет функцию стоимости для линейной регрессии.

    Аргументы:
    X -- матрица признаков (список списков)
    y -- вектор целевых значений (список)
    theta -- вектор параметров (список)

    Возвращает:
    J -- значение функции стоимости
    """
    m = len(y)  # количество примеров
    h = [dot(X[i], theta) for i in range(m)]  # предсказанные значения модели
    errors = [h[i] - y[i] for i in range(m)]  # разница между предсказанным и фактическим значением
    errors_squared = [e ** 2 for e in errors]
    J = (1 / (2 * m)) * sum_list(errors_squared)  # вычисляем J(theta)
    return J


# 4. gradientDescent
def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Выполняет градиентный спуск для нахождения оптимальных параметров theta.

    Аргументы:
    X -- матрица признаков (список списков)
    y -- вектор целевых значений (список)
    theta -- начальный вектор параметров (список)
    alpha -- коэффициент обучения (скорость обучения)
    num_iters -- количество итераций

    Возвращает:
    theta -- обновленный вектор параметров
    J_history -- история изменения функции стоимости на каждой итерации
    """
    m = len(y)  # количество примеров
    J_history = []  # для хранения значения функции стоимости на каждой итерации

    for _ in range(num_iters):
        h = [dot(X[i], theta) for i in range(m)]  # предсказанные значения
        errors = [h[i] - y[i] for i in range(m)]  # ошибки предсказания

        # Обновление параметров theta
        theta_temp = theta.copy()
        for j in range(len(theta)):
            sum_errors = sum_list([errors[i] * X[i][j] for i in range(m)])
            theta_temp[j] = theta[j] - (alpha / m) * sum_errors
        theta = theta_temp

        J_history.append(computeCost(X, y, theta))  # сохраняем значение функции стоимости

    return theta, J_history