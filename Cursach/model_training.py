import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib
from data_processing import load_and_preprocess_data


def train_model(X_train, y_train, preprocessor):
    """
    Обучение модели линейной регрессии.
    """
    # Создание конвейера обучения
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Обучение модели
    model.fit(X_train, y_train)

    # Оценка точности на тренировочных данных
    train_score = model.score(X_train, y_train)
    print(f"Точность модели на тренировочных данных: {train_score:.2f}")

    return model


def save_model(model, filename):
    """
    Сохранение модели в файл.
    """
    joblib.dump(model, filename)
    print(f"Модель сохранена в файл: {filename}")

def load_model(filename):
    """
    Загрузка модели из файла.
    """
    return joblib.load(filename)

def predict_price(model, sample_input):
    """
    Предсказание цены для нового автомобиля.
    """
    # Имена столбцов, ожидаемые моделью
    columns = [
        'brand', 'name', 'bodyType', 'color', 'fuelType',
        'year', 'mileage', 'transmission', 'power',
        'vehicleConfiguration', 'engineName', 'engineDisplacement', 'location'
    ]

    # Преобразование входных данных в DataFrame
    sample_df = pd.DataFrame(sample_input, columns=columns)

    # Предсказание
    return model.predict(sample_df)


if __name__ == '__main__':
    DATA_PATH = '../data/all_regions_trimmed_10000.csv'

    # Загрузка и предобработка данных
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(DATA_PATH)

    # Обучение модели
    model = train_model(X_train, y_train, preprocessor)

    # Сохранение модели в файл
    model_filename = 'car_price_model_10000.pkl'
    save_model(model, model_filename)
