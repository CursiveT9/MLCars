import numpy as np
import time

def scalar_product_loop(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def scalar_product_loop2(a, b):
    return sum(a * b)

def scalar_product_dot(a, b):
    return np.dot(a.T, b)

a = np.array([[1], [2], [3], [4]])
b = np.array([[1], [2], [3], [4]])

# Вывод типов и формы массивов
print(type(a))
print(type(b))

print(a.shape)
print(b.shape)

result_loop = scalar_product_loop(a, b)
result_loop2 = scalar_product_loop2(a, b)

print("Цикл:", result_loop2)

print("По элементно:", result_loop2)

result_dot = scalar_product_dot(a, b)

print("Результат через dot:")
print(result_dot)
