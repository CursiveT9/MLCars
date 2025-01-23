import joblib
import numpy as np

# Загрузка сохранённой модели
model_data = joblib.load('regress_model.pkl')

# Извлечение нужных данных из модели
weights = model_data['model']
category_mappings = model_data['category_mappings']
mean_values = model_data['mean_values']
std_values = model_data['std_values']

# Toyota,Джип 5 дв.,Белый,Бензин,2010.0,170000.0,Вариатор,158.0,1279000,2.0,Салехард
# Kia,Седан,Белый,Бензин,2019.0,91000.0,АКПП,150.0,1980000,2.0,Муравленко
# Лада,Седан,Серебристый,Бензин,2001.0,250000.0,Механика,89.0,80000,1.6,Новый Уренгой

# Новые данные
# new_data = ['Toyota', 'Джип 5 дв.', 'Белый', 'Бензин', 2010.0, 170000.0, 'Вариатор', 158.0, 2.0, 'Салехард']
# new_data = ['Kia', 'Седан', 'Белый', 'Бензин', 2019.0, 91000.0, 'АКПП', 150.0, 2.0, 'Муравленко']
new_data = ['Лада', 'Седан', 'Серебристый', 'Бензин', 2001.0, 250000.0, 'Механика', 89.0, 1.6, 'Новый Уренгой']


# Преобразование категориальных данных с использованием маппинга
new_data_transformed = []

# Индексы категориальных столбцов в new_data
categorical_columns_indices = [0, 1, 2, 3, 6, 9]

# Используем индексы категориальных столбцов для обработки
for i, col in enumerate(new_data):
    if i in categorical_columns_indices:  # Если это категориальный столбец
        # Получаем название столбца по индексу
        column_name = list(category_mappings.keys())[categorical_columns_indices.index(i)]
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

# Линейная регрессия
def predict(X, weights):
    return np.dot(X, weights)

# Предсказание
predicted_price = predict(new_data_transformed, weights)

print(f"Предсказанная цена: {predicted_price[0]:.2f}")
