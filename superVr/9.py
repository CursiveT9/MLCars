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
        return corners, ids
    else:
        return [], []


def overlay_image(background, overlay, corners):
    """Наложение одного изображения на другое с помощью гомографии на основе маркеров."""
    # Преобразование углов маркеров в массив точек
    pts1 = np.array(corners[0], dtype='float32')

    # Предполагаем, что на втором изображении будут те же маркеры, расположенные в аналогичной позиции
    # Наносим позицию на первое изображение
    h, w = overlay.shape[:2]

    # Преобразование углов для второго изображения
    pts2 = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype='float32')

    # Вычисление гомографии между изображениями
    matrix, _ = cv2.findHomography(pts2, pts1)

    # Наложение изображения
    result = cv2.warpPerspective(overlay, matrix, (background.shape[1], background.shape[0]))

    # Маска для наложения
    mask = np.zeros_like(result)
    cv2.fillConvexPoly(mask, np.int32(pts1), (255, 255, 255))

    # Объединение изображений
    masked_result = cv2.bitwise_and(result, mask)
    final_result = cv2.add(background, masked_result)

    return final_result


# === Загрузка изображений ===
image_big = cv2.imread("aruco.jpg")  # Большое изображение
image_small = cv2.imread("aruco2.jpg")  # Маленькое изображение

if image_big is None or image_small is None:
    raise ValueError("Ошибка загрузки изображений. Проверьте пути к файлам.")

# === Обнаружение маркеров ARUCO ===
corners_big, _ = detect_aruco_markers(image_big)
corners_small, _ = detect_aruco_markers(image_small)

if len(corners_big) == 0 or len(corners_small) == 0:
    raise ValueError("Не удалось обнаружить маркеры на одном или обоих изображениях.")

# === Наложение изображения ===
final_image = overlay_image(image_big, image_small, corners_big)

# === Визуализация результата ===
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.title("Наложение меньшего изображения на большее")
plt.axis("off")
plt.show()
