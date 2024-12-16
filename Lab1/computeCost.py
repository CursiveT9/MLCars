def computeCost(x, y, theta):
    m = len(y)  # количество обучающих примеров
    cost = 0  # начальная стоимость

    for i in range(m):
        prediction = theta[0] + theta[1] * x[i]  # предсказание для каждого примера
        error = prediction - y[i]  # ошибка для примера
        cost += error ** 2  # добавление квадрата ошибки

    cost = cost / (2 * m)  # финальная стоимость
    return cost
