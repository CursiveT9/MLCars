import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_aruco_markers(image):
    """Выделяет контуры и маркеры ARUCO на изображении."""
    if len(image.shape) == 3:  # Если изображение цветное
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Пороговое преобразование изображения
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Находим контуры на бинаризованном изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def count_aruco_markers_by_area(image, min_area, max_area):
    """Фильтрует маркеры по площади и возвращает количество маркеров в этом диапазоне."""
    contours = detect_aruco_markers(image)

    valid_markers = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    return len(valid_markers)


def apply_transformations(image):
    """Создаёт 4 варианта изображения с разными параметрами."""
    images = {}

    # 1. Размытие (GaussianBlur)
    images["blurred"] = cv2.GaussianBlur(image, (5, 5), 0)

    # 2. Повышенная яркость/контраст
    alpha, beta = 1.5, 50  # Контраст, Яркость
    images["brightened"] = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 3. Затемнённое изображение
    alpha_dark, beta_dark = 0.7, -30  # Уменьшение контраста и яркости
    images["darkened"] = cv2.convertScaleAbs(image, alpha=alpha_dark, beta=beta_dark)

    # 4. Высокий контраст
    alpha_contrast, beta_contrast = 2.0, 0  # Только увеличение контраста
    images["contrasted"] = cv2.convertScaleAbs(image, alpha=alpha_contrast, beta=beta_contrast)

    return images


# Загружаем изображение с ARUCO
image_path = "333.png"  # Замените на путь к вашему изображению
original_image = cv2.imread(image_path)

# Создаем преобразованные изображения
transformed_images = apply_transformations(original_image)
transformed_images["original"] = original_image  # Добавляем оригинальное изображение

# Диапазон площадей для анализа
area_thresholds = np.arange(100, 2000, 200)

# Создаем график
plt.figure(figsize=(12, 8))

# Для каждого изображения строим график
for name, img in transformed_images.items():
    marker_counts = []

    for min_area in area_thresholds:
        count = count_aruco_markers_by_area(img, min_area, 10000)  # Максимальная площадь может быть очень большой
        marker_counts.append(count)

    plt.plot(area_thresholds, marker_counts, marker='o', label=name)

# Настройки графика
plt.xlabel("Пороговая площадь контуров")
plt.ylabel("Количество маркеров ARUCO")
plt.title("Зависимость числа маркеров ARUCO от пороговой площади для разных преобразований")
plt.legend()
plt.grid(True)
plt.show()

# Отображение всех изображений для наглядности
plt.figure(figsize=(15, 10))

for i, (name, img) in enumerate(transformed_images.items(), 1):
    plt.subplot(2, 3, i)
    if len(img.shape) == 2:  # Если изображение в градациях серого
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.axis('off')

plt.tight_layout()
plt.show()