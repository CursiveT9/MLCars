import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 21)
y_clean = x
noise = np.random.uniform(-2, 2, size=x.shape)
y_noisy = y_clean + noise
theta1_values = np.linspace(-5, 5, 1000)
plt.figure(figsize=(8, 6))

def hypothesis(x, theta1):
    return x * theta1

def error_functional(theta1, x, y):
    return (1/(2*len(x))) * np.sum((hypothesis(x, theta1) - y)**2)

J_values_clean = [error_functional(theta1, x, y_clean) for theta1 in theta1_values]
theta1_min_clean = theta1_values[np.argmin(J_values_clean)]
h_x_min_clean = theta1_min_clean * x

plt.plot(x, y_clean, 'o', color='green', label='Исходные данные y = x')
plt.plot(x, h_x_min_clean, '-', color='green', label=f'{theta1_min_clean:.2f}')

J_values_noisy = [error_functional(theta1, x, y_noisy) for theta1 in theta1_values]
theta1_min_noisy = theta1_values[np.argmin(J_values_noisy)]
h_x_min_noisy = theta1_min_noisy * x

plt.plot(x, y_noisy, 'o', color='red', label='Зашумленные данные y + noise')
plt.plot(x, h_x_min_noisy, '-', color='red', label=f'{theta1_min_noisy:.2f}')

plt.title('Аппроксимирующие прямые для чистых и зашумленных данных')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()