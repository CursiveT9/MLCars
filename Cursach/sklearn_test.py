import joblib
import pandas as pd

# Загрузка сохранённой модели
model = joblib.load('logistic_model_sklearn.pkl')

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

# Преобразование данных в DataFrame
columns = ['brand', 'bodyType', 'color', 'fuelType', 'year', 'mileage', 'transmission', 'power', 'engineDisplacement', 'location']
new_df = pd.DataFrame(data_to_check, columns=columns)

# Предсказание
predicted_classes = model.predict(new_df)
predicted_probabilities = model.predict_proba(new_df)[:, 1]

# Вывод результатов
for i, (car_data, pred_class, pred_prob) in enumerate(zip(data_to_check, predicted_classes, predicted_probabilities)):
    print(f"Автомобиль {i + 1}:")
    print(f"  Данные: {car_data}")
    print(f"  Предсказанный класс: {'Дороже миллиона' if pred_class == 1 else 'Не дороже миллиона'}")
    print(f"  Вероятность: {pred_prob:.2f}")
    print()