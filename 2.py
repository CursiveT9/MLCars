import numpy as np
from matplotlib import pyplot as plt

x = np.arange(1, 21)
y = x  # y_i = x_i

# Значения theta1, которые будем использовать
theta1_values = np.linspace(-4, 4, 50)

noise = np.random.uniform(-2, 2, size=x.shape)
y_noisy = y + noise  # Добавляем шум

# 2. Вычисляем функционал ошибки для каждого theta1 с зашумленными данными
j_noisy_values = []

for theta1 in theta1_values:
    h_x_noisy = theta1 * x  # h(x) = theta1 * x
    j_noisy = np.sum((h_x_noisy - y_noisy) ** 2)  # J(theta1) для зашумленных данных
    j_noisy_values.append(j_noisy)

# 3. Построение графика зависимости J(theta1) от theta1 для зашумленных данных
plt.figure(figsize=(10, 5))
plt.plot(theta1_values, j_noisy_values, label='J(theta1) с шумом', color='orange')
plt.title('Зависимость функционала ошибки J(theta1) от theta1 (зашумленные данные)')
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.grid()
plt.legend()
plt.show()

# 4. Нахождение theta1min для зашумленных данных
theta1_min_noisy = theta1_values[np.argmin(j_noisy_values)]
print(f'Minimum J(theta1) при theta1 (зашумленные данные) = {theta1_min_noisy}')
