import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_and_preprocess_data(file_path):
    """
    Загрузка данных, предобработка и разделение на тренировочный и тестовый наборы.
    """
    # Загрузка данных
    data = pd.read_csv(file_path)

    # Выбор целевой переменной и признаков
    target = 'price'

    # Все доступные признаки, включая категориальные и числовые
    features = [
        'brand', 'name', 'bodyType', 'color', 'fuelType',
        'year', 'mileage', 'transmission', 'power',
        'engineName', 'engineDisplacement', 'location'
    ]

    # Оставляем только нужные столбцы
    data = data[features + [target]]

    # Обработка пустых значений
    data = data.dropna()

    # Разделение на X и y
    X = data[features]
    y = data[target]

    # Определение числовых и категориальных признаков
    numeric_features = ['year', 'mileage', 'power', 'engineDisplacement']
    categorical_features = ['brand', 'name', 'bodyType', 'color', 'fuelType', 'transmission', 'engineName', 'location']

    # Предобработка для числовых данных
    numeric_transformer = StandardScaler()

    # Предобработка для категориальных данных
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Комбинирование трансформеров
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Разделение данных на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor
