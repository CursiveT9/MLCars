import cv2
import numpy as np
import os


def detect_and_draw_markers(image_path, output_path):
    """Обнаруживает и выделяет маркеры ArUCO на изображении"""
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return

    # Инициализация детектора ArUCO
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Детектирование маркеров
    corners, ids, rejected = detector.detectMarkers(image)

    # Рисование обнаруженных маркеров
    if ids is not None:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            # Извлекаем углы маркера
            corners = markerCorner.reshape((4, 2))  # Исправлено: было marerCorner
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # Преобразуем координаты в целые числа
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Рисуем контур маркера
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Вычисляем и рисуем центр маркера
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            # Подписываем ID маркера
            cv2.putText(image, str(markerID),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # Сохраняем результат
    cv2.imwrite(output_path, image)
    print(f"Обработанное изображение сохранено: {output_path}")


def main():
    # Список изображений для обработки
    image_files = {
        'original': 'aruco.jpg',
        'blurred': 'blurred.png',
        'brightened': 'brightened.png',
        'darkened': 'darkened.png',
        'contrasted': 'contrasted.png'
    }

    # Создаем папку для результатов, если ее нет
    os.makedirs('output_markers', exist_ok=True)

    # Обрабатываем каждое изображение
    for name, input_path in image_files.items():
        if os.path.exists(input_path):
            output_path = f"output_markers/markers_{name}.png"
            detect_and_draw_markers(input_path, output_path)
        else:
            print(f"Файл не найден: {input_path}")


if __name__ == "__main__":
    main()
