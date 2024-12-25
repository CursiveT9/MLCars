import methods as m
import matplotlib.pyplot as plt
import numpy as np

# Чтение данных из файла
data = np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:, :2]
y = data[:, 2]

X_normalized, X_mean, X_std = m.z_score_normalization(X)
x1 = X_normalized[:, 0]
x2 = X_normalized[:, 1]

# Добавление нелинейных признаков
X_poly = np.column_stack([
    np.ones(X_normalized.shape[0]),
    x1,
    x2,
    x1 * x2,
    x1 ** 2,
    x2 ** 2
])

# Параметры обучения
alpha = 0.1
epochs = 10000

# Обучение модели
w = m.gradient_descent(X_poly, y, alpha, epochs)
print("Обученные веса:", w)

# Визуализация результатов
x1_vals = np.linspace(X_normalized[:, 0].min(), X_normalized[:, 0].max() + 5, 100)
x2_vals = np.linspace(X_normalized[:, 1].min(), X_normalized[:, 1].max() + 5, 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

# Полиномиальные признаки для визуализации решающей границы
grid_poly = np.column_stack([
    np.ones(x1_grid.ravel().shape),
    x1_grid.ravel(),
    x2_grid.ravel(),
    x1_grid.ravel() * x2_grid.ravel(),
    x1_grid.ravel() ** 2,
    x2_grid.ravel() ** 2
])

z = m.sigmoid(np.dot(grid_poly, w)).reshape(x1_grid.shape)

# Построение графика без тепловой карты, только штриховая линия для разделяющей границы
plt.contour(x1_grid, x2_grid, z, levels=[0.5], colors='black', linewidths=2, linestyles='dashed')

# Наносим точки из нормализованных данных
plt.scatter(X_normalized[y == 0][:, 0], X_normalized[y == 0][:, 1], color="blue", label="Нулевой класс")
plt.scatter(X_normalized[y == 1][:, 0], X_normalized[y == 1][:, 1], color="red", label="Единичный класс")

# Настройка осей и заголовков
plt.xlabel("Вибрация (нормализованная)")
plt.ylabel("Неравномерность вращения (нормализованная)")
plt.title("Разделяющая граница на нормализованных данных")
plt.legend()
plt.show()


# Восстановление исходных значений для сетки
x1_vals_original = np.linspace(X[:, 0].min(), X[:, 0].max() + 50, 100)
x2_vals_original = np.linspace(X[:, 1].min(), X[:, 1].max() + 50, 100)
x1_grid_original, x2_grid_original = np.meshgrid(x1_vals_original, x2_vals_original)

# Нормализация сетки для расчетов
x1_grid_norm = (x1_grid_original - X_mean[0]) / X_std[0]
x2_grid_norm = (x2_grid_original - X_mean[1]) / X_std[1]

# Полиномиальные признаки для нормализованных точек сетки
grid_poly = np.column_stack([
    np.ones(x1_grid_norm.ravel().shape),
    x1_grid_norm.ravel(),
    x2_grid_norm.ravel(),
    x1_grid_norm.ravel() * x2_grid_norm.ravel(),
    x1_grid_norm.ravel() ** 2,
    x2_grid_norm.ravel() ** 2
])

# Вычисление z для разделяющей границы
z = m.sigmoid(np.dot(grid_poly, w)).reshape(x1_grid_original.shape)

# Построение графика
plt.figure(figsize=(8, 6))

# Рисуем разделяющую границу
plt.contour(x1_grid_original, x2_grid_original, z, levels=[0.5], colors='black', linewidths=2, linestyles='dashed')

# Наносим точки из исходного файла (ненормализованные данные)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="blue", label="Нулевой класс")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", label="Единичный класс")

# Настройка осей и заголовков
plt.xlabel("Вибрация")
plt.ylabel("Неравномерность вращения")
plt.title("Разделяющая граница на исходных данных")
plt.legend()
plt.show()

def app():
    print("Введите значения:")
    vibration = float(input("Вибрация: "))
    unevenness = float(input("Неравномерность вращения: "))

    # Нормализация новых данных (Z-score)
    new_data = np.array([[vibration, unevenness]])
    new_data_normalized, _, _ = m.z_score_normalization(new_data)
    x1_new = new_data_normalized[:, 0]
    x2_new = new_data_normalized[:, 1]

    # Создание полиномиальных признаков
    new_data_poly = np.column_stack([
        np.ones(new_data_normalized.shape[0]),
        x1_new,
        x2_new,
        x1_new * x2_new,
        x1_new ** 2,
        x2_new ** 2
    ])

    # Вычисление вероятности неисправности
    probability = m.sigmoid(np.dot(new_data_poly, w))
    print(f"Вероятность неисправности: {probability[0]:.4f}")

app()
