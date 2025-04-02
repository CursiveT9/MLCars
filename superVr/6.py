import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_and_match(img1, img2, num_keypoints):
    """Обнаруживает ключевые точки и выполняет сопоставление изображений."""

    # Инициализируем ORB с заданным числом ключевых точек
    orb = cv2.ORB_create(nfeatures=num_keypoints)

    # Преобразуем в оттенки серого
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Находим ключевые точки и дескрипторы
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Создаем BFMatcher для сопоставления точек
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Находим соответствия между дескрипторами
    matches = bf.match(des1, des2)

    # Сортируем по расстоянию (лучшие совпадения первыми)
    matches = sorted(matches, key=lambda x: x.distance)

    # Найдем минимальное расстояние среди всех совпадений
    min_distance = matches[0].distance if matches else None

    # Отобразим совпадения
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches, min_distance, len(kp1), len(kp2)


# Загружаем изображения (замените на ваши файлы)
image_paths = ["aruco.jpg", "blurred.png", "brightened.png", "darkened.png", "contrasted.png"]
images = [cv2.imread(img) for img in image_paths]

# Проверяем, что изображения загрузились
if any(img is None for img in images):
    raise ValueError("Не удалось загрузить одно или несколько изображений.")

# Параметры количества ключевых точек
num_keypoints_list = [100, 300, 500, 1000]
min_distances = []

# Сравниваем изображения попарно
for num_keypoints in num_keypoints_list:
    print(f"\n=== Количество ключевых точек: {num_keypoints} ===")

    distances = []

    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i + 1]

        img_matches, min_dist, kp1_count, kp2_count = detect_and_match(img1, img2, num_keypoints)

        print(f"Сравнение img{i + 1} и img{i + 2}:")
        print(f"  - Найдено ключевых точек: {kp1_count} и {kp2_count}")
        print(f"  - Минимальное расстояние между дескрипторами: {min_dist:.2f}")

        # Отображаем изображение с совпадениями
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Сопоставление img{i + 1} и img{i + 2} (Ключевых точек: {num_keypoints})")
        plt.axis("off")
        plt.show()

        if min_dist is not None:
            distances.append(min_dist)

    # Среднее минимальное расстояние для данного количества ключевых точек
    if distances:
        min_distances.append(np.mean(distances))
    else:
        min_distances.append(None)

# Построение графика зависимости минимального расстояния от количества ключевых точек
plt.figure(figsize=(8, 5))
plt.plot(num_keypoints_list, min_distances, marker='o', color='b', linestyle='--')
plt.xlabel("Количество ключевых точек")
plt.ylabel("Среднее минимальное расстояние")
plt.title("Влияние количества ключевых точек на качество сопоставления")
plt.grid(True)
plt.show()
