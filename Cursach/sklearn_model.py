import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Загрузка данных
data = pd.read_csv('../data/all_regions_trimmed_400000.csv')

# Создание бинарной целевой переменной (1 — цена > 1 000 000, 0 — цена <= 1 000 000)
data['is_expensive'] = (data['price'] > 1000000).astype(int)

# Отделение признаков от целевой переменной
X = data.drop(columns=['price', 'is_expensive'])
y = data['is_expensive']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Определение категориальных и числовых столбцов
categorical_columns = X.select_dtypes(include=[object]).columns
numeric_columns = X.select_dtypes(include=[np.number]).columns

# Создание пайплайна для предобработки данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),  # Нормализация числовых данных
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # Кодирование категориальных данных
    ]
)

# Создание пайплайна для модели
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', max_iter=30))  # Логистическая регрессия
])

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели (точность)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Сохранение модели и предобработчиков
joblib.dump(model, 'logistic_model_sklearn.pkl')