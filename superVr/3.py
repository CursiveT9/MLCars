import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def count_contours(image, thresholds):
    """Считает количество контуров при разных значениях порога."""
    contour_counts = []

    for thresh in thresholds:
        _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_counts.append(len(contours))

    return contour_counts


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


# Создаем папку для сохранения изображений, если её нет
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Загружаем изображение с ARUCO
image_path = "333.png"  # Замените на путь к вашему изображению
image = cv2.imread(image_path)

# Сохраняем оригинальное изображение
original_path = os.path.join(output_dir, "aruco.jpg")
cv2.imwrite(original_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# Генерируем изменённые версии изображения
images = apply_transformations(image)

# Сохраняем преобразованные изображения
for name, img in images.items():
    output_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(output_path, img)

print("Изображения успешно сохранены в папку:", output_dir)

# Продолжаем обработку для графиков
if len(image.shape) == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = image.copy()

# Диапазон порогов для анализа
thresholds = np.arange(50, 250, 10)

# Подсчёт контуров для разных изображений
plt.figure(figsize=(10, 6))
for name, img in images.items():
    if len(img.shape) == 3:  # Конвертация в серый для пороговой обработки
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contour_counts = count_contours(img, thresholds)
    plt.plot(thresholds, contour_counts, label=name)

# График
plt.xlabel("Пороговый уровень")
plt.ylabel("Количество контуров")
plt.title("Зависимость числа контуров от порогового уровня")
plt.legend()
plt.grid(True)
plt.show()

# Отображаем оригинальное и преобразованные изображения
plt.figure(figsize=(12, 10))

# Оригинальное изображение
plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Оригинал")
plt.axis('off')

# Преобразование 1: Размытие
plt.subplot(3, 2, 2)
plt.imshow(cv2.cvtColor(images["blurred"], cv2.COLOR_BGR2RGB))
plt.title("Размытие (Gaussian Blur)")
plt.axis('off')

# Преобразование 2: Повышенная яркость
plt.subplot(3, 2, 3)
plt.imshow(cv2.cvtColor(images["brightened"], cv2.COLOR_BGR2RGB))
plt.title("Яркость")
plt.axis('off')

# Преобразование 3: Затемнённое
plt.subplot(3, 2, 4)
plt.imshow(cv2.cvtColor(images["darkened"], cv2.COLOR_BGR2RGB))
plt.title("Затемнённое")
plt.axis('off')

# Преобразование 4: Высокий контраст
plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(images["contrasted"], cv2.COLOR_BGR2RGB))
plt.title("Высокий контраст")
plt.axis('off')

# Показываем
plt.tight_layout()
plt.show()