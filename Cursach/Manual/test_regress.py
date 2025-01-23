import joblib
import numpy as np

# Загрузка сохранённой модели
model_data = joblib.load('logistic_model.pkl')

# Извлечение нужных данных из модели
weights = model_data['model']
category_mappings = model_data['category_mappings']
mean_values = model_data['mean_values']
std_values = model_data['std_values']

# Определение функции predict
def predict(X, weights):
    return 1 / (1 + np.exp(-np.dot(X, weights)))  # Сигмоидная функция

# Kia,Хэтчбек 5 дв.,Красный,Бензин,2011.0,236000.0,АКПП,129.0,1077000,1.6,Севастополь
# Mercedes-Benz,Джип 5 дв.,Черный,Дизель,2013.0,207000.0,АКПП,249.0,2450000,3.0,Симферополь
# Лада,Хэтчбек 5 дв.,Серебристый,Бензин,2003.0,90000.0,Механика,77.0,157000,1.5,Симферополь
# Лада,Седан,Серебристый,Бензин,2011.0,160000.0,Механика,81.0,240000,1.6,Советский
# Toyota,Джип 5 дв.,Белый,Бензин,2011.0,170000.0,Вариатор,158.0,1279000,2.0,Салехард
# Kia,Седан,Белый,Бензин,2019.0,91000.0,АКПП,150.0,1980000,2.0,Муравленко
# Лада,Седан,Серебристый,Бензин,2001.0,250000.0,Механика,89.0,80000,1.6,Новый Уренгой

# Данные для проверки
data_to_check = [
    ['Kia', 'Хэтчбек 5 дв.', 'Красный', 'Бензин', 2011.0, 236000.0, 'АКПП', 129.0, 1.6, 'Севастополь'],
    ['Mercedes-Benz', 'Джип 5 дв.', 'Черный', 'Дизель', 2013.0, 207000.0, 'АКПП', 249.0, 3.0, 'Симферополь'],
    ['Лада', 'Хэтчбек 5 дв.', 'Серебристый', 'Бензин', 2003.0, 90000.0, 'Механика', 77.0, 1.5, 'Симферополь'],
    ['Лада', 'Седан', 'Серебристый', 'Бензин', 2011.0, 160000.0, 'Механика', 81.0, 1.6, 'Советский'],
    ['Toyota', 'Джип 5 дв.', 'Белый', 'Бензин', 2011.0, 170000.0, 'Вариатор', 158.0, 2.0, 'Салехард'],
    ['Kia', 'Седан', 'Белый', 'Бензин', 2019.0, 91000.0, 'АКПП', 150.0, 2.0, 'Муравленко'],
    ['Лада', 'Седан', 'Серебристый', 'Бензин', 2001.0, 250000.0, 'Механика', 89.0, 1.6, 'Новый Уренгой']
]

# Индексы категориальных столбцов в данных
categorical_columns_indices = [0, 1, 2, 3, 6, 9]

# Обработка каждой строки данных
for i, new_data in enumerate(data_to_check):
    # Преобразование категориальных данных с использованием маппинга
    new_data_transformed = []
    for j, col in enumerate(new_data):
        if j in categorical_columns_indices:  # Если это категориальный столбец
            # Получаем название столбца по индексу
            column_name = list(category_mappings.keys())[categorical_columns_indices.index(j)]
            if col in category_mappings[column_name]:
                # Если значение найдено в маппинге, заменяем на индекс
                new_data_transformed.append(np.where(category_mappings[column_name] == col)[0][0])
            else:
                # Если значение не найдено в маппинге, используем дефолтное значение (например, -1)
                new_data_transformed.append(-1)  # Можно заменить на любое дефолтное значение
        else:
            # Если это числовой столбец, просто добавляем значение
            new_data_transformed.append(col)

    # Преобразование в числовой тип
    new_data_transformed = np.array(new_data_transformed, dtype=float)

    # Нормализация данных
    new_data_transformed = (new_data_transformed - mean_values) / std_values

    # Убедитесь, что new_data_transformed является массивом numpy
    new_data_transformed = np.array(new_data_transformed).reshape(1, -1)

    # Добавление столбца единиц для смещения (базовый уровень)
    new_data_transformed = np.hstack((np.ones((new_data_transformed.shape[0], 1)), new_data_transformed))

    # Предсказание вероятности
    probability = predict(new_data_transformed, weights)

    # Преобразование вероятности в класс (1 — дороже миллиона, 0 — нет)
    predicted_class = (probability >= 0.5).astype(int)

    # Вывод результата
    print(f"Автомобиль {i + 1}:")
    print(f"  Данные: {new_data}")
    print(f"  Вероятность, что машина дороже миллиона: {probability[0]:.2f}")
    print(f"  Класс: {'Дороже миллиона' if predicted_class[0] == 1 else 'Не дороже миллиона'}")
    print()

