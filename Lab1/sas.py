import numpy as np

def multiply_matrix_by_scalar(matrix, scalar):
    return matrix * scalar

matrix = np.array([
    [1, 2],
    [1, 3],
    [1, 4]
])

matrix2 = np.array([
    [2],
    [3],
    [4]
])

scalar = 2

result1 = multiply_matrix_by_scalar(matrix, scalar)
result2 = multiply_matrix_by_scalar(matrix2, scalar)

result11 = result1 + 5
result22 = result2 + 5

result33 = matrix * matrix2

print(result33)

print(result1)
print(result11)
print(result2)
print(result22)

print(type(result11))
print(type(result22))
