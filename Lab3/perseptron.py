import matplotlib.pyplot as plt
import random
import cvreader

# Путь к изображению
image_path = 'image9.png'

# Создаем объект CVReader
cv_reader = cvreader.CVReader(image_path)

# Находим пересечение осей
origin_x, origin_y = cv_reader.find_axis_intersection()
print(f"Пересечение осей: ({origin_x}, {origin_y})")

# Получаем координаты для синих кругов (нулевой класс)
crosses = cv_reader.get_coordinates(color='blue', step=138)
print(f"Координаты ноликов: {crosses}")

# Получаем координаты для красных крестиков (единичный класс)
naughts = cv_reader.get_coordinates(color='red', step=138)
print(f"Координаты крестиков: {naughts}")

# Добавляем фиктивный столбец x0 = 1
data = [(1, x1, x2, label) for x1, x2, label in crosses + naughts]

# Инициализация весов
weights = [random.uniform(-1, 1) for _ in range(3)]
print(weights)


# Правило Хебба
def train_perceptron(data, weights):
    is_converged = False
    while not is_converged:
        is_converged = True
        for x0, x1, x2, y_true in data:
            z = weights[0] * x0 + weights[1] * x1 + weights[2] * x2
            y_pred = 1 if z >= 0 else 0
            if y_pred != y_true:
                is_converged = False
                if y_pred == 0:
                    weights[0] += x0
                    weights[1] += x1
                    weights[2] += x2
                else:
                    weights[0] -= x0
                    weights[1] -= x1
                    weights[2] -= x2
    return weights


# Обучение
weights = train_perceptron(data, weights)


# Построение разделяющей прямой
def plot_decision_boundary(weights):
    x = [-1.5, 2]  # Диапазон для x1
    y = [-(weights[0] + weights[1] * x_val) / weights[2] for x_val in x]  # Расчет x2
    plt.plot(x, y, 'r-', label='Decision Boundary')


# График
plt.figure(figsize=(8, 6))
for x1, x2, _ in crosses:  # Игнорируем метку класса
    plt.scatter(x1, x2, color='red', label='Crosses' if x1 == crosses[0][0] else "")
for x1, x2, _ in naughts:  # Игнорируем метку класса
    plt.scatter(x1, x2, color='blue', label='Naughts' if x1 == naughts[0][0] else "")
plot_decision_boundary(weights)

plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Добавляем оси
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Включаем сетку
plt.grid(True)

# Добавляем подписи и легенду
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Граница персептрона')

# Показываем график
plt.show()
