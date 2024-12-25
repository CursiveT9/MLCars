import numpy as np
import matplotlib.pyplot as plt

# Параметры модели
phi = [1, 0, 2, 1, 1, 0]
a = 0.5

# Диапазон значений для x1 и x2
x1 = np.linspace(-15, 15, 1000)
x2 = np.linspace(-15, 15, 1000)
x1, x2 = np.meshgrid(x1, x2)

# Уравнение модели
z = phi[0] + phi[1]*x1 + phi[2]*x2 + phi[3]*x1*x2 + phi[4]*x1**2 + phi[5]*x2**2

# Кривая разделения
plt.figure(figsize=(8, 8))
plt.contour(x1, x2, z, levels=[a], colors='black', linestyles='--', linewidths=2)

# Заштрихованные области
class_0 = plt.contourf(x1, x2, z, levels=[-np.inf, a], colors=['violet'], alpha=0.3)
class_1 = plt.contourf(x1, x2, z, levels=[a, np.inf], colors=['red'], alpha=0.3)

# Оформление графика
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Разделяющая кривая и области классов", fontsize=14)
plt.xlabel("$x_1$", fontsize=12)
plt.ylabel("$x_2$", fontsize=12)
plt.grid(True)

plt.xticks(np.arange(-15, 16, 1))
plt.yticks(np.arange(-15, 16, 1))

# Добавление легенды вручную
plt.plot([], [], color='violet', alpha=0.3, label="Нулевой класс")
plt.plot([], [], color='red', alpha=0.3, label="Единичный класс")
plt.legend(loc="upper right")

plt.show()
