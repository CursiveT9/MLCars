import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Создание экспериментальных данных
x = np.arange(1, 21)  # xi = 1, 2, ..., 20
y = x  # yi = xi

# Шаг 2: Определение модели линейной регрессии
theta0 = 0  # theta0 = 0
theta1_values = np.linspace(-4, 4, 50)  # различные значения theta1
errors = []

# Вычисление функционала ошибки J(theta1)
for theta1 in theta1_values:
    h_x = theta0 + theta1 * x  # h(x) = theta0 + theta1 * x
    J_theta1 = np.sum((h_x - y) ** 2)  # сумма квадратов ошибок
    errors.append(J_theta1)

# Шаг 3: Построение графика зависимости J(theta1)
plt.figure(figsize=(10, 6))
plt.plot(theta1_values, errors, label='J(theta1)', color='blue')
plt.title('Зависимость функционала ошибки J(theta1)')
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.grid()
plt.axhline(0, color='gray', lw=0.5, ls='--')
plt.axvline(0, color='gray', lw=0.5, ls='--')
plt.legend()
plt.show()

# Шаг 4: Нахождение минимального theta1
min_index = np.argmin(errors)
theta1_min = theta1_values[min_index]
print(f'Минимальное значение theta1: {theta1_min}')
