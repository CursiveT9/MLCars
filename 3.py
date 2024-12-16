import numpy as np

def scalar_product_loop(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def product_by_element(a, b):
    result = np.array(sum([a[i] * b[i] for i in range(len(a))]))
    return result

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
result_by_element = product_by_element(a, b)
result_dot = scalar_product_dot(a, b)

print("Цикл:", result_loop)
print("По элементно:", result_by_element)
print("Результат через dot:", result_dot)