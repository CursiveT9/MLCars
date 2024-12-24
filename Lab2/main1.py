import numpy as np
import tools

# Загрузка данных
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]  # Входные признаки: обороты двигателя и передачи
y = data[:, 2]    # Целевая переменная: цена
m = len(y)        # Количество примеров

# Добавление единичного столбца для theta_0
X_no_norm = np.c_[np.ones(m), X]

# Параметры для градиентного спуска
alpha = 1e-7  # Малый шаг из-за больших значений в данных
num_iters = 100  # Увеличиваем количество итераций
theta = np.zeros(3)

# Выполнение градиентного спуска без нормализации
theta, J_history = tools.gradientDescent(X_no_norm, y, theta, alpha, num_iters)

# Вывод результатов
print("=== Результаты методом градиентного спуска ===")
print(f"Параметр theta_0 (смещение): {theta[0]:.2f}")
print(f"Параметр theta_1 (вес для оборотов двигателя): {theta[1]:.2f}")
print(f"Параметр theta_2 (вес для количества передач): {theta[2]:.2f}")

# Аналитическое решение (нормальное уравнение)
theta_analytic = tools.normal_equation(X_no_norm, y)

print("\n=== Результаты аналитическим решением ===")
print(f"Параметр theta_0 (смещение): {theta_analytic[0]:.2f}")
print(f"Параметр theta_1 (вес для оборотов двигателя): {theta_analytic[1]:.2f}")
print(f"Параметр theta_2 (вес для количества передач): {theta_analytic[2]:.2f}")

# Сравнение предсказаний
pred_gd = np.dot(X_no_norm, theta)
pred_analytic = np.dot(X_no_norm, theta_analytic)

print("\n=== Сравнение первых пяти предсказаний ===")
for i in range(5):
    print(f"Пример {i + 1}:")
    print(f"  Предсказание (градиентный спуск): {pred_gd[i]:.2f}")
    print(f"  Предсказание (аналитическое решение): {pred_analytic[i]:.2f}")

# График сходимости функции стоимости
tools.plot_cost_function(J_history)

# График сравнения предсказаний
tools.plot_predictions_comparison(y, X, pred_gd, pred_analytic)

# График сравнения предсказаний с наложением
tools.plot_predictions_comparison2(y, X, pred_gd, pred_analytic)
