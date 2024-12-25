import numpy as np
import matplotlib.pyplot as plt

# Параметры модели
phi = [1, 0, 2, 1, 1, 0]  # Коэффициенты уравнения (веса модели)
a = 0.5  # Уровень разделяющей кривой

# Диапазон значений для x1 и x2
x1 = np.linspace(-15, 15, 1000)  # Значения для первой переменной (x1)
x2 = np.linspace(-15, 15, 1000)  # Значения для второй переменной (x2)
x1, x2 = np.meshgrid(x1, x2)  # Создаём сетку точек для визуализации

# Уравнение модели
# Вычисляем значения функции z для каждой точки сетки (x1, x2)
z = phi[0] + phi[1]*x1 + phi[2]*x2 + phi[3]*x1*x2 + phi[4]*x1**2 + phi[5]*x2**2

# Кривая разделения
plt.figure(figsize=(8, 8))  # Устанавливаем размер графика
plt.contour(x1, x2, z, levels=[a], colors='black', linestyles='--', linewidths=2)
# Рисуем разделяющую кривую на уровне z = a

# Заштрихованные области
class_0 = plt.contourf(x1, x2, z, levels=[-np.inf, a], colors=['violet'], alpha=0.3)
# Заштриховываем область, где z < a (нулевой класс)
class_1 = plt.contourf(x1, x2, z, levels=[a, np.inf], colors=['red'], alpha=0.3)
# Заштриховываем область, где z >= a (единичный класс)

# Оформление графика
plt.axhline(0, color='black', linewidth=0.8)  # Добавляем горизонтальную ось
plt.axvline(0, color='black', linewidth=0.8)  # Добавляем вертикальную ось
plt.title("Разделяющая кривая и области классов", fontsize=14)  # Заголовок графика
plt.xlabel("$x_1$", fontsize=12)  # Подпись оси x
plt.ylabel("$x_2$", fontsize=12)  # Подпись оси y
plt.grid(True)  # Добавляем сетку для удобства чтения

plt.xticks(np.arange(-15, 16, 1))  # Устанавливаем шаг значений по оси x
plt.yticks(np.arange(-15, 16, 1))  # Устанавливаем шаг значений по оси y

# Добавление легенды вручную
plt.plot([], [], color='violet', alpha=0.3, label="Нулевой класс")  # Нулевой класс
plt.plot([], [], color='red', alpha=0.3, label="Единичный класс")  # Единичный класс
plt.legend(loc="upper right")  # Позиционируем легенду в правом верхнем углу

plt.show()  # Отображаем график
