import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Функция сигмоиды
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Предотвращение переполнения
    return 1 / (1 + np.exp(-z))

# Функция вычисления стоимости
def compute_cost(X, y, w):
    m = len(y)
    z = np.dot(X, w)
    predictions = sigmoid(z)
    epsilon = 1e-15  # Для избежания логарифма от 0
    cost = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return cost

# Функция вычисления градиента
def compute_gradient(X, y, w):
    m = len(y)
    z = np.dot(X, w)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y)) / m
    return gradient

# Градиентный спуск
def gradient_descent(X, y, alpha, epochs):
    w = np.zeros(X.shape[1])  # Инициализация весов нулями
    for epoch in range(epochs):
        gradient = compute_gradient(X, y, w)
        w -= alpha * gradient
        if epoch % 1000 == 0 or epoch == epochs - 1:  # Лог каждые 1000 эпох
            cost = compute_cost(X, y, w)
            print(f"Эпоха - {epoch}, значение функции - {cost:.4f}")
    return w

# Чтение данных из файла
data = np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:, :2]
y = data[:, 2]

# Нормализация данных (Z-score)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std
x1 = X_normalized[:, 0]
x2 = X_normalized[:, 1]

# Добавление полиномиальных признаков
X_poly = np.column_stack([
    np.ones(X_normalized.shape[0]),
    x1,
    x2,
    x1 * x2
])

# Создание фигуры и двух подграфиков
plt.figure(figsize=(12, 6))

# Подграфик для исходных данных
plt.subplot(1, 2, 1)  # 1 строка, 2 столбца, 1-й график
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="green", label="Нулевой класс")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", label="Единичный класс")
plt.xlabel("Вибрация (исходные данные)")
plt.ylabel("Неравномерность вращения (исходные данные)")
plt.title("Исходные данные")
plt.legend()
plt.grid(alpha=0.3)

# Подграфик для нормализованных данных
plt.subplot(1, 2, 2)  # 1 строка, 2 столбца, 2-й график
plt.scatter(X_normalized[y == 0][:, 0], X_normalized[y == 0][:, 1], color="green", label="Нулевой класс")
plt.scatter(X_normalized[y == 1][:, 0], X_normalized[y == 1][:, 1], color="red", label="Единичный класс")
plt.xlabel("Вибрация (нормализованные данные)")
plt.ylabel("Неравномерность вращения (нормализованные данные)")
plt.title("Нормализованные данные")
plt.legend()
plt.grid(alpha=0.3)

# Отображение графиков
plt.tight_layout()  # Уменьшает наложение текста и графиков
plt.show()

# Параметры обучения
alpha = 0.1
epochs = 10000

# Обучение модели
w = gradient_descent(X_poly, y, alpha, epochs)
print("Обученные веса:", w)

# Визуализация данных
x1_vals = np.linspace(X_normalized[:, 0].min(), X_normalized[:, 0].max(), 100)
x2_vals = np.linspace(X_normalized[:, 1].min(), X_normalized[:, 1].max(), 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

# Полиномиальные признаки для визуализации решающей границы
grid_poly = np.column_stack([
    np.ones(x1_grid.ravel().shape),
    x1_grid.ravel(),
    x2_grid.ravel(),
    x1_grid.ravel() * x2_grid.ravel()
])

z = sigmoid(np.dot(grid_poly, w)).reshape(x1_grid.shape)

padding = 2.0  # Дополнительное пространство вокруг данных
x1_vals = np.linspace(X_normalized[:, 0].min() - padding, X_normalized[:, 0].max() + padding, 100)
x2_vals = np.linspace(X_normalized[:, 1].min() - padding, X_normalized[:, 1].max() + padding, 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

# Полиномиальные признаки для визуализации решающей границы
grid_poly = np.column_stack([
    np.ones(x1_grid.ravel().shape),
    x1_grid.ravel(),
    x2_grid.ravel(),
    x1_grid.ravel() * x2_grid.ravel()
])

# Вычисление значений для кривой
z = sigmoid(np.dot(grid_poly, w)).reshape(x1_grid.shape)

# Построение графика
plt.figure(figsize=(8, 6))
cmap = mcolors.LinearSegmentedColormap.from_list("RedGreen", ["green", "red"])
plt.contourf(x1_grid, x2_grid, z, levels=50, cmap=cmap, alpha=0.8)
plt.colorbar(label="P(y=1)")
plt.scatter(X_normalized[y == 0][:, 0], X_normalized[y == 0][:, 1], color="white", label="Нулевой класс")
plt.scatter(X_normalized[y == 1][:, 0], X_normalized[y == 1][:, 1], color="black", label="Единичный класс")

# Увеличение области графика
plt.xlim(x1_vals.min(), x1_vals.max())
plt.ylim(x2_vals.min(), x2_vals.max())

# Настройки подписей и заголовков
plt.xlabel("Вибрация")
plt.ylabel("Неравномерность вращения")
plt.title("Вероятность неисправности двигателя")
plt.legend()

# Отображение графика
plt.show()

# Функция тестирования новых данных
def test():
    print("\nВведите значения:")
    vibration = float(input("Вибрация: "))
    unevenness = float(input("Неравномерность вращения: "))

    # Нормализация новых данных (Z-score)
    new_data = np.array([[vibration, unevenness]])
    new_data_normalized = (new_data - X_mean) / X_std
    x1_new = new_data_normalized[:, 0]
    x2_new = new_data_normalized[:, 1]

    # Создание полиномиальных признаков
    new_data_poly = np.column_stack([
        np.ones(new_data_normalized.shape[0]),
        x1_new,
        x2_new,
        x1_new * x2_new
    ])

    # Вычисление вероятности неисправности
    probability = sigmoid(np.dot(new_data_poly, w))
    print(f"Прогноз неисправности: {probability[0]:.0f}")

test()
