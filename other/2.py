import numpy as np
import matplotlib.pyplot as plt

# Данные
x_data = np.arange(1, 21)  # Создаём массив значений x от 1 до 20
y_clean = x_data  # Чистые данные: y = x (линейная зависимость без шума)
noise = np.random.uniform(-2, 2, size=x_data.shape)  # Генерируем шум в диапазоне [-2, 2] для каждого x
y_noisy = y_clean + noise  # Добавляем шум к данным y, чтобы получить зашумлённые данные

# Значения theta1 для анализа
theta1_values = np.linspace(-5, 5, 1000)  # Массив значений theta1 от -5 до 5 (1000 точек)

# Гипотеза
def hypothesis(x, theta1):
    """
    Вычисляет значения гипотезы h(x) = theta1 * x.
    """
    return theta1 * x

# Функционал ошибки
def error_functional_with_theta0_zero(theta1, x, y):
    """
    Вычисляет функционал ошибки J(theta1) для заданных x и y, при условии theta0 = 0.
    Формула:
    J(theta1) = 1 / (2 * m) * sum((h(x) - y)^2), где m - количество точек.
    """
    return (1 / (2 * len(x))) * np.sum((hypothesis(x, theta1) - y) ** 2)

# Вычисление ошибок для чистых данных
J_theta1_clean = [error_functional_with_theta0_zero(theta1, x_data, y_clean) for theta1 in theta1_values]
# Ищем значение theta1, при котором ошибка минимальна (чистые данные)
theta1_min_clean = theta1_values[np.argmin(J_theta1_clean)]  # np.argmin возвращает индекс минимума

# Вычисление ошибок для зашумлённых данных
J_theta1_noisy = [error_functional_with_theta0_zero(theta1, x_data, y_noisy) for theta1 in theta1_values]
# Ищем значение theta1, при котором ошибка минимальна (зашумлённые данные)
theta1_min_noisy = theta1_values[np.argmin(J_theta1_noisy)]

# Построение графиков
fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # Создаём два графика, расположенных вертикально

# Первый график: аппроксимирующие прямые
h_x_min_clean = theta1_min_clean * x_data  # Вычисляем h(x) для чистых данных и оптимального theta1
h_x_min_noisy = theta1_min_noisy * x_data  # Вычисляем h(x) для зашумлённых данных и оптимального theta1

# Наносим чистые данные и аппроксимирующую прямую
axes[0].plot(x_data, y_clean, 'o', color='green', label='Исходные данные y = x')  # Чистые данные
axes[0].plot(x_data, h_x_min_clean, '-', color='green', label=f'h(x) для чистых данных (theta1={theta1_min_clean:.2f})')

# Наносим зашумлённые данные и аппроксимирующую прямую
axes[0].plot(x_data, y_noisy, 'o', color='red', label='Зашумлённые данные y + noise')  # Зашумлённые данные
axes[0].plot(x_data, h_x_min_noisy, '-', color='red', label=f'h(x) для зашумлённых данных (theta1={theta1_min_noisy:.2f})')

# Настройки для первого графика
axes[0].set_title('Аппроксимирующие прямые для чистых и зашумлённых данных')  # Заголовок
axes[0].set_xlabel('x')  # Подпись оси x
axes[0].set_ylabel('y')  # Подпись оси y
axes[0].legend()  # Отображение легенды
axes[0].grid(True)  # Сетка на графике

# Второй график: функционал ошибки
axes[1].plot(theta1_values, J_theta1_clean, label="Чистые данные J(theta1)", color='blue')  # Ошибки для чистых данных
axes[1].axvline(x=theta1_min_clean, color='red', linestyle='--', label=f'Минимум θ1 (чистые)={theta1_min_clean:.2f}')
# Вертикальная линия на минимуме для чистых данных

axes[1].plot(theta1_values, J_theta1_noisy, label="Зашумленные данные J(theta1)", color='green')  # Ошибки для зашумлённых данных
axes[1].axvline(x=theta1_min_noisy, color='yellow', linestyle='--', label=f'Минимум θ1 (зашумленные)={theta1_min_noisy:.2f}')
# Вертикальная линия на минимуме для зашумлённых данных

# Настройки для второго графика
axes[1].set_xlabel('theta1')  # Подпись оси x
axes[1].set_ylabel('J(theta1)')  # Подпись оси y
axes[1].set_title('Зависимость ошибки J(theta1) для чистых и зашумленных данных')  # Заголовок
axes[1].legend()  # Отображение легенды
axes[1].grid(True)  # Сетка на графике
axes[1].set_ylim([-5, 10])  # Устанавливаем диапазон значений по оси y
axes[1].set_xlim([0, 2])  # Устанавливаем диапазон значений по оси x

# Сохранение и отображение графика
plt.tight_layout()  # Уменьшение отступов между графиками
plt.savefig('combined_plots.png', dpi=300)  # Сохраняем график в файл
plt.show()  # Отображаем график


# Имеем экспериментальные данные  xi = 1,2...,20    yi = xi .
# 1) Построить аппроксимирующую прямую  h(x) = theta0 + theta1 x ,считая theta0 = 0, для разных значений theta1
# 2) Построить зависимость функционала ошибки J(theta1) =  1/(2m)*sum(h(xi)-yi)^2  от b
# и из графика найти theta1=theta1min, соответствующее минимуму функционала
# Выполнить предыдущее задание для зашумлённых данных (- добавить равномерно распределённые случайные данные в диапазоне (-2, 2))
