import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_aruco_markers(image):
    """Обнаруживает ArUco маркеры и возвращает их угловые координаты."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        all_corners = np.vstack(corners).reshape(-1, 2)  # Объединяем все точки
        x_min, y_min = np.min(all_corners, axis=0)
        x_max, y_max = np.max(all_corners, axis=0)
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    else:
        return None


def detect_keypoints(image, num_keypoints=500):
    """Обнаруживает ключевые точки с помощью ORB."""
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def filter_keypoints_by_bbox(keypoints, descriptors, bbox):
    """Оставляет только те ключевые точки, которые находятся внутри bbox."""
    x_min, y_min, x_max, y_max = bbox
    filtered_keypoints = []
    filtered_descriptors = []

    for kp, desc in zip(keypoints, descriptors):
        x, y = kp.pt
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_keypoints.append(kp)
            filtered_descriptors.append(desc)

    return filtered_keypoints, np.array(filtered_descriptors)


def match_and_draw(img1, img2, kp1, des1, kp2, des2):
    """Сопоставляет ключевые точки и отображает совпадения."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches, matches


# === Загрузка изображений ===
img1 = cv2.imread("aruco.jpg")  # Оригинальное изображение
img2 = cv2.imread("aruco2.jpg")  # Измененное изображение

if img1 is None or img2 is None:
    raise ValueError("Ошибка загрузки изображений. Проверьте пути к файлам.")

# === Поиск ArUco маркеров ===
bbox = detect_aruco_markers(img1)
if bbox is None:
    raise ValueError("Не удалось обнаружить ArUco маркеры!")

# === Обнаружение ключевых точек ===
kp1, des1 = detect_keypoints(img1)
kp2, des2 = detect_keypoints(img2)

# === Фильтрация ключевых точек по ArUco bounding box ===
kp1_filtered, des1_filtered = filter_keypoints_by_bbox(kp1, des1, bbox)
kp2_filtered, des2_filtered = filter_keypoints_by_bbox(kp2, des2, bbox)

print(f"Ключевые точки внутри ArUco области: {len(kp1_filtered)} и {len(kp2_filtered)}")

# === Сопоставление точек ===
if len(kp1_filtered) > 0 and len(kp2_filtered) > 0:
    img_matches, matches = match_and_draw(img1, img2, kp1_filtered, des1_filtered, kp2_filtered, des2_filtered)

    # === Визуализация результатов ===
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("Сопоставление изображений с учетом ArUco bounding box")
    plt.axis("off")
    plt.show()

    # Выводим среднее и минимальное расстояние
    # distances = [m.distance for m in matches]
    distances = [m.distance for m in matches if m.distance > 1]

    print(f"Среднее расстояние между дескрипторами: {np.mean(distances):.2f}")
    print(f"Минимальное расстояние: {np.min(distances):.2f}")
else:
    print("Недостаточно ключевых точек после фильтрации для сопоставления.")

# === График зависимости числа ключевых точек от их точности ===
num_keypoints_list = [50, 100, 200, 300, 500]
avg_distances = []
min_distances = []

for num_keypoints in num_keypoints_list:
    kp1, des1 = detect_keypoints(img1, num_keypoints)
    kp2, des2 = detect_keypoints(img2, num_keypoints)

    kp1_filtered, des1_filtered = filter_keypoints_by_bbox(kp1, des1, bbox)
    kp2_filtered, des2_filtered = filter_keypoints_by_bbox(kp2, des2, bbox)

    if len(kp1_filtered) > 0 and len(kp2_filtered) > 0:
        _, matches = match_and_draw(img1, img2, kp1_filtered, des1_filtered, kp2_filtered, des2_filtered)
        distances = [m.distance for m in matches]

        avg_distances.append(np.mean(distances))
        min_distances.append(np.min(distances))
    else:
        avg_distances.append(None)
        min_distances.append(None)

plt.figure(figsize=(10, 5))
plt.plot(num_keypoints_list, avg_distances, marker='o', linestyle='-', color='b', label="Среднее расстояние")
plt.plot(num_keypoints_list, min_distances, marker='s', linestyle='--', color='r', label="Минимальное расстояние")
plt.xlabel("Количество ключевых точек")
plt.ylabel("Расстояние между дескрипторами")
plt.title("Влияние количества ключевых точек на точность сопоставления (ArUco ограничение)")
plt.legend()
plt.grid(True)
plt.show()
