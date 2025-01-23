import pandas as pd

# Путь к исходному файлу
input_file_path = '../data/all_regions.csv'

# Путь для сохранения нового файла
output_file_path = '../data/all_regions_trimmed_400000.csv'

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

# Удаление строк с пустыми полями
df_cleaned = df.dropna()

# Сохранение первых n строк в новый файл
df_cleaned.head(400000).to_csv(output_file_path, index=False)

print(f"Файл с миллионом строк без пустых полей сохранен как {output_file_path}.")
