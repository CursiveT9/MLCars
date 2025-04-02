import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.arange(1, 21)
y = x

def h(x, theta1):
    return theta1 * x

def compute_cost(x, y, theta1):
    m = len(y)
    predictions = h(x, theta1)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

theta1_vals = np.linspace(0, 2, 100000)
cost_vals = []

for theta1 in theta1_vals:
    cost_vals.append(compute_cost(x, y, theta1))

# Часть 2: Добавляем шум в данные
noise = np.random.uniform(-2, 2, size=y.shape)
y_noisy = y + noise

cost_vals_noisy = []
for theta1 in theta1_vals:
    cost_vals_noisy.append(compute_cost(x, y_noisy, theta1))

theta1_min = theta1_vals[np.argmin(cost_vals)]
theta1_min_noisy = theta1_vals[np.argmin(cost_vals_noisy)]

print(f'Минимальное значение theta1 для чистых данных: {theta1_min}')
print(f'Минимальное значение theta1 для зашумленных данных: {theta1_min_noisy}')

plt.figure(figsize=(10, 6))

plt.plot(theta1_vals, cost_vals, color='blue', label='Чистые данные')
plt.plot(theta1_vals, cost_vals_noisy, color='orange', label='Зашумленные данные')

plt.scatter(theta1_min, np.min(cost_vals), color='blue', marker='x', s=100, label=f'Минимум (чистые данные) $\\theta_1$={theta1_min:.2f}')
plt.scatter(theta1_min_noisy, np.min(cost_vals_noisy), color='orange', marker='x', s=100, label=f'Минимум (зашумленные данные) $\\theta_1$={theta1_min_noisy:.2f}')

plt.xlabel(r'$\theta_1$')
plt.ylabel('J($\\theta_1$)')
plt.title('Зависимость функционала ошибки от $\\theta_1$')
plt.legend()
plt.grid(True)

plt.show()
