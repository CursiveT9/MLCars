# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from math import atan2, degrees
#
#
# def detect_keypoints(image, num_keypoints=500):
#     """Обнаруживает ключевые точки с помощью ORB."""
#     orb = cv2.ORB_create(nfeatures=num_keypoints)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     keypoints, descriptors = orb.detectAndCompute(gray, None)
#     return keypoints, descriptors
#
#
# def match_and_draw(img1, img2, kp1, des1, kp2, des2):
#     """Сопоставляет ключевые точки и отображает совпадения."""
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     matches = sorted(matches, key=lambda x: x.distance)
#
#     img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None,
#                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     return img_matches, matches
#
#
# def line_angle(p1, p2):
#     """Вычисляет угол наклона линии, соединяющей две точки p1 и p2."""
#     x1, y1 = p1
#     x2, y2 = p2
#     return atan2(y2 - y1, x2 - x1)
#
#
# def are_lines_parallel(angle1, angle2, tolerance=5.0):
#     """Проверяет, параллельны ли две линии с заданной точностью."""
#     return abs(degrees(angle1 - angle2)) < tolerance
#
#
# # === Загрузка изображений ===
# img1 = cv2.imread("aruco.jpg")  # Оригинальное изображение
# img2 = cv2.imread("aruco2.jpg")  # Измененное изображение
#
# if img1 is None or img2 is None:
#     raise ValueError("Ошибка загрузки изображений. Проверьте пути к файлам.")
#
# # === Обнаружение ключевых точек ===
# kp1, des1 = detect_keypoints(img1)
# kp2, des2 = detect_keypoints(img2)
#
# # === Сопоставление ключевых точек ===
# img_matches, matches = match_and_draw(img1, img2, kp1, des1, kp2, des2)
#
# # === Проверка параллельности линий ===
# parallel_pairs = []
# for m in matches:
#     # Индексы ключевых точек
#     idx1 = m.queryIdx
#     idx2 = m.trainIdx
#
#     # Получаем координаты ключевых точек
#     p1 = kp1[idx1].pt
#     p2 = kp2[idx2].pt
#
#     # Вычисляем углы наклона линий
#     angle1 = line_angle(kp1[0].pt, p1)
#     angle2 = line_angle(kp2[0].pt, p2)
#
#     # Проверяем параллельность
#     if are_lines_parallel(angle1, angle2):
#         parallel_pairs.append((p1, p2))
#
# # === Визуализация результатов ===
# for p1, p2 in parallel_pairs:
#     cv2.line(img1, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)
#     cv2.line(img2, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)
#
# # Отображаем изображение с параллельными линиями
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# plt.title("Оригинальное изображение с параллельными линиями")
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# plt.title("Измененное изображение с параллельными линиями")
# plt.axis('off')
#
# plt.show()
#
# # Печать количества найденных параллельных линий
# print(f"Количество параллельных линий: {len(parallel_pairs)}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees


def detect_keypoints(image, num_keypoints=500):
    """Обнаруживает ключевые точки с помощью ORB."""
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_and_draw(img1, img2, kp1, des1, kp2, des2):
    """Сопоставляет ключевые точки и отображает совпадения."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches, matches


def line_angle(p1, p2):
    """Вычисляет угол наклона линии, соединяющей две точки p1 и p2."""
    x1, y1 = p1
    x2, y2 = p2
    return atan2(y2 - y1, x2 - x1)


def are_lines_parallel(angle1, angle2, tolerance=5.0):
    """Проверяет, параллельны ли две линии с заданной точностью."""
    return abs(degrees(angle1 - angle2)) < tolerance


def extend_line(p1, p2, length_factor=2):
    """Удлиняет линию, соединяющую p1 и p2, на заданный множитель длины."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # Удлиняем точку p2 в направлении линии на указанный множитель
    extended_x2 = x2 + dx * length_factor
    extended_y2 = y2 + dy * length_factor

    # Для точки p1 наоборот — уменьшаем расстояние
    extended_x1 = x1 - dx * length_factor
    extended_y1 = y1 - dy * length_factor

    return (extended_x1, extended_y1), (extended_x2, extended_y2)


# === Загрузка изображений ===
img1 = cv2.imread("aruco.jpg")  # Оригинальное изображение
img2 = cv2.imread("aruco2.jpg")  # Измененное изображение

if img1 is None or img2 is None:
    raise ValueError("Ошибка загрузки изображений. Проверьте пути к файлам.")

# === Обнаружение ключевых точек ===
kp1, des1 = detect_keypoints(img1)
kp2, des2 = detect_keypoints(img2)

# === Сопоставление ключевых точек ===
img_matches, matches = match_and_draw(img1, img2, kp1, des1, kp2, des2)

# === Проверка параллельности линий ===
parallel_pairs = []
for m in matches:
    # Индексы ключевых точек
    idx1 = m.queryIdx
    idx2 = m.trainIdx

    # Получаем координаты ключевых точек
    p1 = kp1[idx1].pt
    p2 = kp2[idx2].pt

    # Вычисляем углы наклона линий
    angle1 = line_angle(kp1[0].pt, p1)
    angle2 = line_angle(kp2[0].pt, p2)

    # Проверяем параллельность
    if are_lines_parallel(angle1, angle2):
        parallel_pairs.append((p1, p2))

# === Визуализация результатов с удлиненными линиями ===
for p1, p2 in parallel_pairs:
    extended_p1, extended_p2 = extend_line(p1, p2, length_factor=2)  # Удлиняем линию
    cv2.line(img1, tuple(map(int, extended_p1)), tuple(map(int, extended_p2)), (0, 255, 0), 2)
    cv2.line(img2, tuple(map(int, extended_p1)), tuple(map(int, extended_p2)), (0, 255, 0), 2)

# Отображаем изображение с параллельными линиями
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title("Оригинальное изображение с удлиненными параллельными линиями")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title("Измененное изображение с удлиненными параллельными линиями")
plt.axis('off')

plt.show()

# Печать количества найденных параллельных линий
print(f"Количество параллельных линий: {len(parallel_pairs)}")
