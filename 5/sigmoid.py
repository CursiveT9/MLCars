import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def exp(x, terms=130):
    result = np.float64(0)  # Убедитесь, что result - это тип float64
    for n in range(terms):
        result += np.float64((x ** n) / factorial(n))  # Явное приведение к float64
    return result

# def exp(x, terms=130):
#     result = 0
#     for n in range(terms):
#         result += (x ** n) / factorial(n)
#     return result

def sigmoid(x):
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Гиперболический синус
def sinh(x):
    return (exp(x) - exp(-x)) / 2

# Гиперболический косинус
def cosh(x):
    return (exp(x) + exp(-x)) / 2

# Гиперболический тангенс
def tanh(x):
    return sinh(x) / cosh(x)

def tanh_derivative(x):
    return 1 - tanh(x)**2

x_values = [0, 3, -3, 8, -8, 15, -15]

# Вычисление значений сигмоиды с точностью до 15 знаков
sigmoid_values = [f"{sigmoid(x):.15f}" for x in x_values]

print("Значения сигмоиды в точках:")
for x, val in zip(x_values, sigmoid_values):
    print(f"sigmoid({x}) = {val}")

x = np.linspace(-5, 5, 400)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(x), label="Сигмоида", color='red')
plt.plot(x, sigmoid_derivative(x), label="Производная сигмоиды", linestyle='--', color='blue')
plt.title("Сигмойда и его производная")
plt.legend()
plt.grid(True)
plt.ylim(-1.5, 1.5)
plt.xticks(np.arange(-5, 6, 1))
plt.yticks(np.arange(-1.5, 2, 0.5))

plt.subplot(1, 2, 2)
plt.plot(x, sinh(x), label="Sinh", color='green')
plt.plot(x, cosh(x), label="Cosh", color='brown')
plt.plot(x, tanh(x), label="Tanh", color='violet')
plt.plot(x, tanh_derivative(x), label="Производная tanh", linestyle='--', color='blue')
plt.title("Гиперболического тангенс и его производная")
plt.legend()
plt.grid(True)
plt.ylim(-5, 5)
plt.xticks(np.arange(-7, 8, 1))
plt.yticks(np.arange(-5, 6, 1))

plt.tight_layout()
plt.show()

#
# import numpy as np
# import matplotlib.pyplot as plt
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def sigmoid_derivative(x):
#     sig = sigmoid(x)
#     return sig * (1 - sig)
#
# # Гиперболический синус
# def sinh(x):
#     return (np.exp(x) - np.exp(-x)) / 2
#
# # Гиперболический косинус
# def cosh(x):
#     return (np.exp(x) + np.exp(-x)) / 2
#
# # Гиперболический тангенс
# def tanh(x):
#     return sinh(x) / cosh(x)
#
# def tanh_derivative(x):
#     return 1 - tanh(x)**2
#
# x_values = [0, 3, -3, 8, -8, 15, -15]
#
# # Вычисление значений сигмоиды с точностью до 15 знаков
# sigmoid_values = [f"{sigmoid(x):.15f}" for x in x_values]
#
# print("Значения сигмоиды в точках:")
# for x, val in zip(x_values, sigmoid_values):
#     print(f"sigmoid({x}) = {val}")
#
# x = np.linspace(-5, 5, 400)
#
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.plot(x, sigmoid(x), label="Сигмоида", color='red')
# plt.plot(x, sigmoid_derivative(x), label="Производная сигмоиды", linestyle='--', color='blue')
# plt.title("Сигмойда и его производная")
# plt.legend()
# plt.grid(True)
# plt.ylim(-1.5, 1.5)
# plt.xticks(np.arange(-5, 6, 1))
# plt.yticks(np.arange(-1.5, 2, 0.5))
#
# plt.subplot(1, 2, 2)
# plt.plot(x, sinh(x), label="Sinh", color='green')
# plt.plot(x, cosh(x), label="Cosh", color='brown')
# plt.plot(x, tanh(x), label="Tanh", color='violet')
# plt.plot(x, tanh_derivative(x), label="Производная tanh", linestyle='--', color='blue')
# plt.title("Гиперболического тангенс и его производная")
# plt.legend()
# plt.grid(True)
# plt.ylim(-5, 5)
# plt.xticks(np.arange(-7, 8, 1))
# plt.yticks(np.arange(-5, 6, 1))
#
# plt.tight_layout()
# plt.show()
#
