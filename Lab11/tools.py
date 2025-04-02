def eye(n):
    """
    Создает единичную матрицу размером n x n.
    """
    identity_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        identity_matrix.append(row)
    return identity_matrix

def zeros(shape):
    """
    Создает матрицу или вектор из нулей заданной формы.
    """
    if isinstance(shape, int):
        # Вектор длины shape
        return [0 for _ in range(shape)]
    elif isinstance(shape, tuple) and len(shape) == 2:
        rows, cols = shape
        return [[0 for _ in range(cols)] for _ in range(rows)]
    else:
        raise ValueError("Shape must be int or tuple of length 2")

def ones(n):
    """
    Создает вектор из единиц длины n.
    """
    return [1 for _ in range(n)]

def dot(a, b):
    """
    Вычисляет скалярное произведение двух векторов или умножение матрицы на вектор.
    """
    if isinstance(a[0], list):
        # a — матрица
        result = []
        for row in a:
            sum_prod = 0
            for i in range(len(row)):
                sum_prod += row[i] * b[i]
            result.append(sum_prod)
        return result
    else:
        # a и b — векторы
        return sum([a[i] * b[i] for i in range(len(a))])

def sum_list(a):
    """
    Вычисляет сумму элементов вектора или матрицы.
    """
    total = 0
    if isinstance(a[0], list):
        # Матрица
        for row in a:
            total += sum(row)
    else:
        # Вектор
        total = sum(a)
    return total

def transpose(a):
    """
    Вычисляет транспонированную матрицу.
    """
    transposed = []
    for i in range(len(a[0])):
        transposed_row = []
        for row in a:
            transposed_row.append(row[i])
        transposed.append(transposed_row)
    return transposed

def column_stack(arrays):
    """
    Объединяет 1D-массивы как столбцы в 2D-массив.
    """
    stacked = []
    for i in range(len(arrays[0])):
        row = []
        for array in arrays:
            row.append(array[i])
        stacked.append(row)
    return stacked

def loadtxt(filename, delimiter=','):
    """
    Загружает данные из текстового файла.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line_data = line.strip().split(delimiter)
            data.append([float(item) for item in line_data])
    return data
