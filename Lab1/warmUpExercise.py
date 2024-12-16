import numpy as np

# Функция для создания единичной матрицы
def create_identity_matrix(n):
    # Способ 1: С использованием стандартных функций
    identity_matrix_standard = np.eye(n)

    # Способ 2: Без использования стандартных функций
    identity_matrix_custom = np.zeros((n, n))  # Создаём матрицу размерности n x n, заполненную нулями
    for i in range(n):
        identity_matrix_custom[i][i] = 1  # Заполняем диагональ единицами

    return identity_matrix_standard, identity_matrix_custom


# Ввод размерности матрицы
n = int(input("Введите размерность матрицы n: "))

# Создание матрицы
identity_matrix_standard, identity_matrix_custom = create_identity_matrix(n)

# Вывод результатов
print("\nЕдиничная матрица, созданная с использованием стандартных функций:")
print(identity_matrix_standard)

print("\nЕдиничная матрица, созданная без использования стандартных функций:")
print(identity_matrix_custom)
