import numpy as np
import matplotlib.pyplot as plt
import tools

# Загрузка данных
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]  # Входные признаки: обороты двигателя и передачи
y = data[:, 2]    # Целевая переменная: цена
m = len(y)        # Количество примеров

# Нормализация данных
X_norm, mu, sigma = tools.feature_normalize(X)
X_norm = np.c_[np.ones(m), X_norm]  # Добавляем единичный столбец для theta_0

# Список различных значений альфы
alphas = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0]
alphas2 = [0.01, 0.1, 0.3, 0.5, 1.0]

# Сохранение истории стоимости для разных альф
J_histories1 = []
J_histories2 = []

# Параметры для градиентного спуска
num_iters = 10

# Для первого набора значений альфы выполняем градиентный спуск
for alpha in alphas:
    theta = np.zeros(3)  # Инициализация параметров
    theta, J_history = tools.gradientDescent(X_norm, y, theta, alpha, num_iters)
    J_histories1.append(J_history)

# Для второго набора значений альфы выполняем градиентный спуск
for alpha in alphas2:
    theta = np.zeros(3)  # Инициализация параметров
    theta, J_history = tools.gradientDescent(X_norm, y, theta, alpha, num_iters)
    J_histories2.append(J_history)

# Построение графиков сходимости для разных альф
plt.figure(figsize=(12, 8))

# График для первого набора альф
plt.subplot(2, 1, 1)
for i, alpha in enumerate(alphas):
    plt.plot(range(num_iters), J_histories1[i], label=f'alpha = {alpha}')
plt.yscale('log')
plt.xlabel('Итерации')
plt.ylabel('Функция стоимости J (log scale)')
plt.title('Сходимость функции стоимости для различных значений alpha - 1-й график')
plt.legend()
plt.grid(True)

# График для второго набора альф
plt.subplot(2, 1, 2)
for i, alpha in enumerate(alphas2):
    plt.plot(range(num_iters), J_histories2[i], label=f'alpha = {alpha}')
plt.yscale('log')
plt.xlabel('Итерации')
plt.ylabel('Функция стоимости J (log scale)')
plt.title('Сходимость функции стоимости для различных значений alpha - 2-й график')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
