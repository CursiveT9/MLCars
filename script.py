import pandas as pd

# Путь к исходному файлу
input_file_path = 'data/all_regions.csv'

# Путь для сохранения нового файла
output_file_path = 'data/all_regions_trimmed_1000000.csv'

# Загрузка всего CSV файла
df = pd.read_csv(input_file_path)

# Удаление строк с пустыми полями
df_cleaned = df.dropna()

# Сохранение первых миллион строк в новый файл
df_cleaned.head(1000000).to_csv(output_file_path, index=False)

print(f"Файл с миллионом строк без пустых полей сохранен как {output_file_path}.")
