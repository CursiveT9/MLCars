import pandas as pd

# Путь к исходному файлу
input_file_path = '../data/all_regions.csv'

# Путь для сохранения нового файла (без пустых полей)
output_file_path = '../data/all_regions_trimmed_400000.csv'

# Путь для сохранения файла с пустыми полями
output_file_path_missing = '../data/all_regions_missing_rows.csv'

# Загрузка всего CSV файла
df = pd.read_csv(input_file_path)

# Удаление ненужных столбцов
columns_to_drop = ['link', 'description', 'parse_date', 'date', 'vehicleConfiguration', 'name', 'engineName']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Преобразование столбца engineDisplacement
if 'engineDisplacement' in df.columns:
    df['engineDisplacement'] = (
        df['engineDisplacement']
        .str.replace(' LTR', '', regex=False)  # Удаление ' LTR'
        .astype(float)  # Преобразование в float
    )

# Удаление строк с пустыми полями и сохранение их в отдельный DataFrame
df_cleaned = df.dropna()  # Строки без пустых полей
df_missing = df[df.isna().any(axis=1)]  # Строки с пустыми полями

# Сохранение первых 400 000 строк без пустых полей в новый файл
df_cleaned.head(400000).to_csv(output_file_path, index=False)

# Сохранение строк с пустыми полями в отдельный файл
df_missing.to_csv(output_file_path_missing, index=False)

print(f"Файл с 400 000 строк без пустых полей сохранен как {output_file_path}.")
print(f"Файл с строками, содержащими пустые поля, сохранен как {output_file_path_missing}.")