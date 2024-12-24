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
    result = 0
    for n in range(terms):
        result += (x ** n) / factorial(n)
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

x = np.linspace(-7, 7, 400)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(x), label="Сигмоида", color='blue')
plt.plot(x, sigmoid_derivative(x), label="Производная сигмоиды", linestyle='--', color='orange')
plt.title("Сигмойда и его производная")
plt.legend()
plt.grid(True)
plt.ylim(-1.1, 1.1)
plt.xticks(np.arange(-7, 8, 1))

plt.subplot(1, 2, 2)
plt.plot(x, sinh(x), label="Sinh", color='red')
plt.plot(x, cosh(x), label="Cosh", color='blue')
plt.plot(x, tanh(x), label="Tanh", color='green')
plt.plot(x, tanh_derivative(x), label="Произв. tanh", linestyle='--', color='red')
plt.title("Гиперболического тангенс и его производная")
plt.legend()
plt.grid(True)
plt.ylim(-5, 5)
plt.xticks(np.arange(-7, 8, 1))

plt.tight_layout()
plt.show()


























